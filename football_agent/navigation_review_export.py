'''Attach later reviewer proposals to exact action prefixes, evaluator-only.'''
from copy import deepcopy

from .inquiry_contract import utc

VERSION = 'navigation-review-export-v1'


def attach_review_proposals(rows, proposals, *, as_of):
    '''Preserve the acquisition row; add only time-available provisional reviews.

Each proposal names an exact reviewed prefix. Similar-looking later states are
not silently relabelled; missing reviews and missing task truth remain missing.
'''
    cutoff, by_prefix = utc(as_of), {}
    for proposal in proposals:
        key = proposal['prefix_event_hash']
        if key in by_prefix: raise ValueError('Ambiguous duplicate prefix review')
        review = proposal['review']
        if review['case_id'] != proposal['case_id'] or review['review_status'] != 'PROVISIONAL_REQUIRES_ADJUDICATION':
            raise ValueError('Review identity/status mismatch')
        if (review['task_outcome_label'] is not None or review['world_truth_verified'] is not False
                or review['eligible_for_value_calibration'] is not False):
            raise ValueError('Provisional review must not assert task truth or value calibration')
        if utc(proposal['available_at']) < utc(proposal['evidence_cutoff']):
            raise ValueError('Review cannot predate its evidence cutoff')
        by_prefix[key] = deepcopy(proposal)
    output = []
    for row in rows:
        if 'evaluator_review' in row: raise ValueError('Preserve prior export; do not overwrite annotations')
        annotations = {}
        for side in ('before', 'after'):
            proposal = by_prefix.get(row[side + '_event_hash'])
            annotations[side] = proposal if proposal and utc(proposal['available_at']) <= cutoff else None
        changed = None
        if all(annotations.values()):
            before = {x['criterion_id']: x['status'] for x in annotations['before']['review']['criteria']}
            after = {x['criterion_id']: x['status'] for x in annotations['after']['review']['criteria']}
            changed = sorted(key for key in before if before[key] != after[key])
        output.append({**deepcopy(row), 'evaluator_review': {
            'schema_version': VERSION, 'export_as_of': cutoff.isoformat(), **annotations,
            'criterion_statuses_changed': changed, 'loss_delta': None,
            'cross_packet_fixture_alignment': 'not_independently_adjudicated',
            'criterion_change_is_utility_change': False,
            'scope': 'later_provisional_review_not_actor_input_or_causal_value',
            'task_truth_available': False, 'eligible_for_value_calibration': False}})
    return output
