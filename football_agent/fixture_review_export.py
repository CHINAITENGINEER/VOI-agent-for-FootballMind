'''Time-censored fixture-bound reviews attached only to evaluator output.'''
from .fixture_bound_review import VERSION, compare_bound_reviews
from .navigation_review_export import attach_review_proposals


def attach_fixture_reviews(rows, proposals, *, as_of):
    for proposal in proposals:
        if proposal['review']['schema_version'] != VERSION:
            raise ValueError('Unbound legacy review cannot supply fixture alignment')
    output = attach_review_proposals(rows, proposals, as_of=as_of)
    for row in output:
        annotation = row['evaluator_review']
        before, after = annotation['before'], annotation['after']
        comparison = compare_bound_reviews(before['review'], after['review']) if before and after else None
        annotation.update(schema_version='fixture-review-export-v1', target_comparison=comparison,
            criterion_statuses_changed=comparison['criterion_statuses_changed'] if comparison else None,
            cross_packet_fixture_alignment=comparison['alignment'] if comparison else 'missing_prefix_review',
            scope='later_fixture_bound_provisional_review_not_actor_input_or_causal_value')
    return output
