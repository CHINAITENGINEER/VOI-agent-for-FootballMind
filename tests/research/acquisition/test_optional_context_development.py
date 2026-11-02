"""Constructed retrospective evaluation contracts, not real performance data."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import socket

import pytest

from football_research.evaluation.optional_context_development import evaluate_context_development
from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor
from test_context_predictor import WORDS, visible

CUTOFF = '2025-01-15T00:00:00+00:00'
AS_OF = '2025-01-25T00:00:00+00:00'
LABELS = [0]*7+[1]*3+[2]*2


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*a, **k):
        raise AssertionError('Development assessment must be offline')
    monkeypatch.setattr(socket, 'getaddrinfo', denied)
    monkeypatch.setattr(socket.socket, 'connect', denied)


def fixture(physical, *, day, y, domain='missing', text=None, run=None):
    run = run or physical+'-run'
    data = []
    for index, body in enumerate((None, text)):
        state = visible(physical+'-provider', day=day, text=body)
        state['before']['decision_at'] = (datetime(2025, 1, day, 10, 0, 1+index,
                                                tzinfo=timezone.utc)).isoformat()
        if domain == 'missing':
            state['before']['prior_probabilities'] = None
            state['before']['prior_basis'] = 'not_supplied_no_prior_inferred_from_current_prediction'
        data.append({'physical_match_id': physical, 'run_id': run, 'prefix_index': index,
            'purpose': 'f_train', 'visible_state': state, 'outcome': y,
            'result_available_at': None if y is None else f'2025-01-{day:02}T22:00:00+00:00'})
    return data


def data(domain='missing'):
    return [row for i, y in enumerate(LABELS)
        for row in fixture(f'A:{i}', day=i+1, y=y, domain=domain, text=WORDS[y])]


def evaluate(records):
    return evaluate_context_development(records, training_cutoff=CUTOFF, as_of=AS_OF,
        evidence_origin='constructed_mechanism', dataset_version='constructed-development-contract-v1')


def test_six_matches_and_alias_repetition_do_not_fit_or_create_scores(monkeypatch):
    train = data()[:12]
    for i, row in enumerate(train):
        row['outcome'] = [1, 1, 2, 2, 2, 2][i//2]
    aliases = deepcopy(train)
    for row in aliases:
        row['run_id'] += '-alias'
        row['visible_state']['before']['match_id'] += '-alias'
    monkeypatch.setattr(OptionalPriorContextPredictor, 'fit', lambda *a, **k: pytest.fail('No fit allowed'))
    result = evaluate(train*8+aliases+fixture('D:1', day=20, y=2, text=WORDS[0]))
    assert result['status'] == 'not_ready' and result['model'] is result['summary'] is None
    assert result['per_match'] == [] and result['actual_fit_execution_at'] is None
    assert result['readiness']['physical_matches'] == 6
    assert result['readiness']['class_counts'] == {'0': 0, '1': 2, '2': 4}
    assert result['readiness']['distinct_prefixes'] == 12
    assert result['duplicate_run_prefix_rows'] == 84


def test_actual_decision_boundary_and_label_time_purge_complete_physical_fixtures():
    train = data()
    for row in train[-2:]:
        row['result_available_at'] = '2025-01-16T00:00:00+00:00'
    cross = fixture('cross', day=20, y=0, text=WORDS[0])
    for index, row in enumerate(cross):
        row['visible_state'] = visible('cross-provider', day=14+2*index,
                                      text=None if not index else WORDS[0])
        row['visible_state']['before']['prior_probabilities'] = None
        row['visible_state']['before']['prior_basis'] = 'not_supplied_no_prior_inferred_from_current_prediction'
        row['visible_state']['before']['fixture']['kickoff'] = '2025-01-20T19:00:00+00:00'
        row['visible_state']['before']['prediction_cutoff'] = '2025-01-20T18:00:00+00:00'
    result = evaluate(train+cross+fixture('pending', day=21, y=None, text=WORDS[1]))
    excluded = {row['physical_match_id']: row['reasons'] for row in result['split']['excluded']}
    assert excluded['A:11'] == ['independent_settlement_not_available_at_training_cutoff']
    assert 'physical_fixture_crosses_actual_decision_boundary' in excluded['cross']
    assert excluded['pending'] == ['independent_settlement_pending_for_some_or_all_prefixes']
    assert result['readiness']['physical_matches'] == 11 and result['status'] == 'not_ready'
    assert result['split']['development_physical_match_ids'] == []


def test_same_f_base_pairing_equal_fixture_weights_negative_results_and_actual_fit_clock():
    original = data()+fixture('D:1', day=20, y=2, text=WORDS[0])+fixture('D:2', day=21, y=0, text=WORDS[2])
    repeated = deepcopy(original[-2:])
    for row in repeated:
        row['run_id'] += '-later'
        row['visible_state']['before']['match_id'] += '-alias'
        now = datetime.fromisoformat(row['visible_state']['before']['decision_at'])
        row['visible_state']['before']['decision_at'] = (now+timedelta(hours=1)).isoformat()
    full = original+repeated
    before = deepcopy(full)
    result = evaluate(full)
    assert full == before and result['status'] == 'evaluated'
    assert result['scope'] == 'retrospective_A_development' and not result['independent_confirmation']
    assert result['model']['metadata']['empirical_base']['probabilities'] == [7/12, 3/12, 2/12]
    assert datetime.fromisoformat(result['actual_fit_execution_at']) > datetime.fromisoformat(AS_OF)
    assert result['model']['metadata']['training_as_of'] == CUTOFF
    assert result['model']['metadata']['training_physical_match_ids'] == sorted(f'A:{i}' for i in range(12))
    assert result['per_match'][1]['other_run_ids_not_scored'] == ['D:2-run-later']
    for position in ('initial', 'terminal'):
        table = result['summary'][position]['all_forecasts_including_explicit_fallbacks']
        assert table['physical_matches'] == 2
        actual = []
        for row in result['per_match']:
            p = row[position]['forecast']['probabilities']
            expected = {'log_loss': -math.log(p[row['outcome']]),
                        'brier_score': sum((v-int(i == row['outcome']))**2 for i, v in enumerate(p))}
            assert row[position]['losses']['forecast'] == pytest.approx(expected)
            actual.append(expected)
        assert table['forecast'] == pytest.approx({key: sum(x[key] for x in actual)/2 for key in actual[0]})
    assert result['summary']['terminal']['supported_context_comparisons_only']['base_minus_forecast']['log_loss'] < 0
    assert not result['information_value_established'] and not result['production_ready']
    json.dumps(result, allow_nan=False)


def test_unsupported_missing_domain_is_explicit_fallback_not_known_context_gain():
    result = evaluate(data('provided')+fixture('D:1', day=20, y=2, text=WORDS[0]))
    terminal = result['per_match'][0]['terminal']
    assert terminal['forecast']['probabilities'] == [7/12, 3/12, 2/12]
    assert terminal['forecast']['prior_probabilities'] is None
    assert terminal['forecast']['fallback_reason'] == 'missing_prior_context_unsupported'
    assert terminal['context_gain'] is None and not terminal['context_comparison_eligible']
    table = result['summary']['terminal']
    assert table['supported_context_comparisons_only'] is None
    assert table['domain_unsupported_physical_matches'] == table['fallback_physical_matches'] == 1
    assert table['all_forecasts_including_explicit_fallbacks']['forecast'] == table['all_forecasts_including_explicit_fallbacks']['base']


def test_exact_body_title_repost_overlap_is_disclosed_without_removing_difficult_records():
    heldout = fixture('D:1', day=20, y=2, text=WORDS[0])
    doc = heldout[1]['visible_state']['before']['documents'][0]
    doc['content_refs'].append({**doc['content_refs'][0], 'role': 'headline'})
    result = evaluate(data()+heldout)
    overlap = result['duplication']
    assert overlap['shared_text_count'] == 1 and overlap['rows_removed_for_text_overlap'] == 0
    assert overlap['development']['same_text_body_and_title_or_anchor'] == 1
    assert overlap['shared_texts'][0]['development_roles'] == ['body', 'headline']
    assert result['summary']['terminal']['all_forecasts_including_explicit_fallbacks']['physical_matches'] == 1


def test_heldout_targets_and_unread_candidates_do_not_fit_base_or_change_forecast():
    rows = data()+fixture('D:1', day=20, y=2, text=WORDS[0])
    first = evaluate(rows)
    changed = deepcopy(rows)
    for row in changed[-2:]:
        row['outcome'] = 1
        row['visible_state']['candidates'] = {'unread': {'body': WORDS[1], 'future_result': 1}}
    second = evaluate(changed)
    assert first['model']['metadata']['training_records_sha256'] == second['model']['metadata']['training_records_sha256']
    assert first['model']['metadata']['empirical_base'] == second['model']['metadata']['empirical_base']
    for position in ('initial', 'terminal'):
        assert first['per_match'][0][position]['forecast']['probabilities'] == second['per_match'][0][position]['forecast']['probabilities']
    assert first['per_match'][0]['terminal']['losses'] != second['per_match'][0]['terminal']['losses']


@pytest.mark.parametrize('damage,reason', [
    (lambda r: r[0].update(purpose='calibration'), 'A/f_train'),
    (lambda r: r[-1]['visible_state']['before'].update(match_id='A:0-provider'), 'provider identity'),
    (lambda r: r[-1].update(prefix_index=4), 'contiguous'),
    (lambda r: r[-1]['visible_state']['before']['documents'][0].update(available_at='2025-01-22T00:00:00Z'), 'clock'),
])
def test_phase_identity_incomplete_history_and_future_content_are_rejected(damage, reason):
    rows = data()+fixture('D:1', day=20, y=2, text=WORDS[0])
    damage(rows)
    with pytest.raises(ValueError, match=reason):
        evaluate(rows)


def test_no_development_never_spends_a_fit_or_publishes_a_model(monkeypatch):
    monkeypatch.setattr(OptionalPriorContextPredictor, 'fit', lambda *a, **k: pytest.fail('No fit allowed'))
    result = evaluate(data())
    assert result['status'] == 'no_development_matches' and result['readiness']['ready_to_fit']
    assert result['model'] is result['summary'] is None
