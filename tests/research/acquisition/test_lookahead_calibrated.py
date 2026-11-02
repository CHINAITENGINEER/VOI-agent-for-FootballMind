from football_research.acquisition.lookahead_calibrated import select_two_step_v2
from football_research.acquisition.policy import select_or_stop


def test_negative_first_step_can_unlock_positive_pair_without_peeking():
    # This directory is intentionally not a Python package. Import the local
    # fixture by its module name so the test works both from the repository
    # root and when the acquisition suite is run directly.
    from test_risk_calibrated import _state, _action
    from datetime import datetime, timezone
    state = _state(datetime(2025,3,1,tzinfo=timezone.utc), 'fixture')
    actions = [_action('elo'),_action('xg')]
    calls = []

    class Estimator:
        def estimate_value(self, visible, candidates, **kwargs):
            # Constructed complementarity: first reveal costs more than its
            # immediate gain; the second is valuable conditional on the first.
            observed = any(g['status']=='observed' for g in visible['groups'].values())
            gain = .1 if observed else 0.
            return [{'action_id':a['action_id'],'action':a,'gain':gain,'lower_gain':gain,
                     'support_matches':100,'version':'constructed_pair'} for a in candidates]

    def predictor(visible):
        calls.append(visible)
        assert all(g['status']=='not_acquired' for g in visible['groups'].values())
        return {'probabilities':[.4,.3,.3]}

    estimator = Estimator()
    assert select_or_stop(state, estimator.estimate_value(state,actions), {'money':.02,'requests':2}, {'money':1})['decision']=='stop'
    result = select_two_step_v2(state,actions,predictor=predictor,estimator=estimator,
                               budget={'money':.02,'requests':2},lambda_cost={'money':1})
    assert result['decision']=='action'
    assert result['net_gain'] > 0
    assert len(calls)==1
    assert result['two_step'][0]['first_net_gain'] < 0
    short = select_two_step_v2(state,actions,predictor=predictor,estimator=estimator,
                               budget={'money':.01,'requests':1},lambda_cost={'money':1})
    assert short['decision']=='stop'
