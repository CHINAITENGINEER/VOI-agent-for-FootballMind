from copy import deepcopy
import pytest
from football_research.acquisition import historical_gain_head as raw
from football_research.acquisition import state_candidate_gain as g1
from football_research.acquisition import policy_return_continuation_g as h


def setup():
    q = g1.build_query({'schema': raw.QUERY_SCHEMA, 'probabilities': [.4, .3, .3],
        'acquired_documents': 0, 'visible_candidates': 2, 'article_age_hours': 0., 'title': 'team fitness'},
        {'physical_match_id': 'fixture', 'used_input_ids': [], 'claims': []})
    base = g1.fit([{'physical_match_id': mid, 'query': q, 'signed_gain': .01} for mid in ('a', 'b')])
    menu = [{'action_id': i, 'query': deepcopy(q)} for i in ('a', 'b')]
    packet = {'candidate': menu[0], 'menu': menu, 'cost_loss_units': .001}
    return base, menu, packet


def test_teacher_choice_is_independent_of_realized_returns():
    base, menu, _ = setup()
    edges = [{**a, 'signed_gain': -.03 if a['action_id'] == 'a' else 100.} for a in menu]
    first = h.successor_target(base, edges, .001)
    assert first['selected_action_id'] == 'a'
    assert first['continuation_target'] == pytest.approx(-.031)
    edges[0]['signed_gain'] = .5; edges[1]['signed_gain'] = -1000.
    assert h.successor_target(base, edges, .001)['selected_action_id'] == 'a'


def test_stop_and_unselected_returns_are_not_read():
    base, menu, _ = setup()
    edges = [{**a, 'signed_gain': object()} for a in menu]
    assert h.successor_target(base, edges, .1)['continuation_target'] == 0
    edges[0]['signed_gain'] = -.03
    assert h.successor_target(base, edges, 0)['continuation_target'] == -.03
    assert h.successor_target(base, [], 0)['selected_action_id'] is None


def training_rows(packet):
    return [{'physical_match_id': mid, 'teacher_fit_match_ids': [other],
             'packet': deepcopy(packet), 'continuation_target': -.02}
            for mid, other in [('a', 'b'), ('b', 'a')]]


def test_negative_targets_are_fitted_and_roundtrip(tmp_path):
    base, menu, packet = setup()
    model = h.fit(training_rows(packet), base)
    assert model['intercept'] == pytest.approx(-.02)
    pred = h.estimate(model, packet, remaining_reads=2)
    assert pred['raw_continuation_estimate'] == pytest.approx(-.02)
    assert pred['continuation_net_value'] == 0
    path = tmp_path/'h.json'; h.save(model, path)
    assert h.choose(model, menu, 2, .001) == h.choose(h.load(path), menu, 2, .001)


def test_whole_match_guard():
    base, _, packet = setup(); rows = training_rows(packet)
    rows[0]['teacher_fit_match_ids'] = ['a', 'b']
    with pytest.raises(ValueError, match='exclude whole'): h.fit(rows, base)


@pytest.mark.parametrize('hidden', ['outcome', 'next_state', 'body', 'prediction_after'])
def test_inference_rejects_hidden_data(hidden):
    base, _, packet = setup(); model = h.fit(training_rows(packet), base)
    packet[hidden] = 'hidden'
    with pytest.raises(ValueError, match='contract'): h.estimate(model, packet, remaining_reads=2)


def test_horizon_one_and_input_immutability():
    base, _, packet = setup(); rows = training_rows(packet)
    for r in rows: r['continuation_target'] = .02
    model = h.fit(rows, base); saved = deepcopy(model)
    assert h.estimate(model, packet, remaining_reads=1)['continuation_net_value'] == 0
    assert h.estimate(model, packet, remaining_reads=2)['continuation_net_value'] == pytest.approx(.02)
    assert model == saved
