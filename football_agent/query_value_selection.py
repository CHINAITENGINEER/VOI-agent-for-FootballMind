"""Shared empirical-menu selection; unknown is not a zero-valued observation."""


def predictor_value_blocker(prediction):
    """Only a predictor-produced capability declaration can block numeric value.

    Candidate titles cannot prove what an unread article contains. This check
    therefore never infers zero information value from a title or a zero delta.
    """
    support = prediction.get('predictor_support')
    if support is None:
        return None  # Preserve the archived policies' original contract.
    if (support.get('schema') not in {'semantic-fixed-f-support-v1', 'entity-semantic-fixed-f-support-v1'}
            or support.get('model_version') != prediction.get('model_version')
            or any(support.get(k) not in {'responsive', 'insensitive'} for k in
                   ('body_response_status', 'title_response_status', 'text_response_status'))):
        raise ValueError('Predictor support must bind the actual frozen f prediction')
    # A search/read can also reveal previously unknown titles. Body blindness
    # alone is therefore insufficient to block a title-sensitive predictor.
    return ('predictor_cannot_resolve_query_value'
            if support['text_response_status'] == 'insensitive' else None)


def select_query_candidates(candidates, *, candidate_generation_complete):
    """Select on current supported means, never certify global sufficiency.

    Kept byte-compatible with the existing context/historical policies. The
    execution runtime separately owns the bounded UNKNOWN exploration budget.
    """
    feasible = [c for c in candidates if c['feasible']]
    positive = [c for c in feasible
                if c['value']['status'] == 'COMPUTED' and c['value']['net_voi'] > 1e-12]
    selection = {'candidate_id': None, 'stop_certificate': None}
    if positive:
        best = min(positive, key=lambda c: (-c['value']['net_voi'], c['action_id']))
        selection.update(decision='acquire', reason='positive_empirical_context_gain',
                         candidate_id=best['action_id'])
    elif not feasible and any(c['reason'] == 'physical_budget_exhausted' for c in candidates):
        selection.update(decision='halt', reason='physical_budget_exhausted')
    elif any(c['value']['status'] != 'COMPUTED' for c in feasible):
        selection.update(decision='defer', reason='context_gain_support_unknown')
    elif not candidate_generation_complete:
        selection.update(decision='defer', reason='candidate_generation_incomplete')
    else:
        selection.update(decision='stop', reason='declared_menu_nonpositive_empirical_mean')
    return selection
