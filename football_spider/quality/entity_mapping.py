"""Entity mapping quality derived from normalized source contracts."""
from __future__ import annotations


def entity_mapping_state(batch, issues=()):
    """Classify source-level entity usability without database access."""
    if not batch.matches:
        return 'not_evaluated'
    qualities = []
    for match in batch.matches:
        qualities.extend((match.competition.id_quality, match.home.id_quality, match.away.id_quality))
    if 'match_scoped' in qualities:
        return 'unmapped'
    if any(value in {'source_name', 'unknown'} for value in qualities):
        return 'ambiguous'
    for issue in issues:
        rule = issue.rule
        if 'CONFLICT' in rule or 'AMBIG' in rule:
            return 'ambiguous'
        if 'UNRESOLVED' in rule or 'IDENTITY' in rule:
            return 'unmapped'
    return 'mapped'


__all__ = ['entity_mapping_state']
