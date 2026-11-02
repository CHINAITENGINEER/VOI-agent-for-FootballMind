"""Bounded, source-attributed evidence updates for historical f research.

This does not infer ground truth or first public availability. Only a later
same-publisher, different-document resolved assertion supersedes a pending
assertion for the exact same source entity and facet. Contradictions remain.
"""
from copy import deepcopy
from datetime import datetime
from collections import defaultdict
from .context_predictor import _digest
from .entity_semantic_f import normalize_entity
from . import scoped_reviewed_f as scoped
from . import reviewed_rich_f as rich

VERSION = 'entity-reviewed-evidence-updates-v1'
PENDING_SELECTION = frozenset({'possible_bench', 'club_eligibility_pending'})
CONFIRMED_SELECTION = scoped.SELECTION


def timestamp(value):
    if value is None:
        return None
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('Timezone-aware evidence timestamp required')
    return t


def status(event):
    e = event['evidence']; atoms = e['atoms']
    concept = scoped.field(atoms, 'concept:')
    predicate = scoped.field(atoms, 'predicate:')
    scope = scoped.field(atoms, 'scope:')
    if concept in PENDING_SELECTION | CONFIRMED_SELECTION:
        return 'selection', concept, concept in PENDING_SELECTION, scope
    if predicate and predicate.startswith('availability:'):
        value = predicate.split(':', 1)[1]
        return 'availability', value, value == 'doubtful', scope
    # Training and medical states are independent facets, never mutually exclusive.
    return f'context:{concept or predicate}', concept or predicate, False, scope


def resolve(row):
    cutoff = timestamp(row['decision_at'])
    if cutoff is None:
        raise ValueError('Decision cutoff required')
    events = row['evidence_events']
    ids = [e['event_id'] for e in events]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate evidence event ID')
    groups = defaultdict(list); audit = []
    for e in events:
        if e['physical_match_id'] != row['physical_match_id']:
            raise ValueError('Evidence fixture binding differs')
        published = timestamp(e['source_published_at'])
        if published is not None and published > cutoff:
            raise ValueError('Future evidence cannot enter this state')
        facet, value, pending, scope = status(e)
        groups[(e['evidence']['team'], normalize_entity(e['entity']), facet)].append(e)
    superseded = {}; conflicts = []
    for key, group in groups.items():
        for old in group:
            facet, value, pending, scope = status(old)
            if not pending or key[0] not in ('home', 'away') or scope not in ('this_match', 'team_context'):
                continue
            candidates = []
            for new in group:
                nf, nv, npending, ns = status(new)
                resolved = (nf == 'selection' and nv in CONFIRMED_SELECTION and ns == 'this_match') or (
                    nf == 'availability' and nv in ('available', 'unavailable') and ns == scope == 'this_match')
                ot, nt = timestamp(old['source_published_at']), timestamp(new['source_published_at'])
                if (resolved and old['source_host'] and old['source_host'] == new['source_host'] and
                        old['document_url'] != new['document_url'] and ot is not None and nt is not None and nt > ot):
                    candidates.append(new)
            # Multiple incompatible resolved assertions are not silently adjudicated.
            if candidates and len({status(e)[1] for e in candidates}) == 1:
                superseded[old['event_id']] = sorted(e['event_id'] for e in candidates)
        active = [e for e in group if e['event_id'] not in superseded]
        values = {status(e)[1] for e in active}
        if key[2] in ('selection', 'availability') and len(values) > 1:
            conflicts.append({'team': key[0], 'entity': key[1], 'facet': key[2],
                              'event_ids': sorted(e['event_id'] for e in active),
                              'status': 'unresolved_source_disagreement'})
    kept = []
    for e in events:
        replacements = superseded.get(e['event_id'], [])
        audit.append({'event_id': e['event_id'], 'input_id': e['input_id'], 'entity': e['entity'],
                      'facet': status(e)[0], 'source_published_at': e['source_published_at'],
                      'status': 'superseded_pending_assertion' if replacements else 'retained',
                      'superseded_by': replacements, 'information_value': None})
        if not replacements:
            kept.append(e['evidence'])
    resolved_row = {**deepcopy(row), 'evidence': rich.unique_evidence(kept)}
    return resolved_row, {'schema': VERSION, 'events': audit, 'conflicts': conflicts,
                          'time_basis': 'declared_source_publication_not_proven_historical_availability',
                          'source_truth_independently_verified': False}


def fit(rows, regularization):
    return {'schema': VERSION, 'head': scoped.fit([resolve(r)[0] for r in rows], regularization),
            'production_ready': False, 'g_training': False}


def predict(model, row):
    if model.get('schema') != VERSION:
        raise ValueError('Temporal model contract differs')
    return scoped.predict(model['head'], resolve(row)[0])


def save(model, path):
    rich.save(model, path)


def load(path):
    import json
    from pathlib import Path
    value = json.loads(Path(path).read_text(encoding='utf-8')); digest = value.pop('model_sha256')
    if value.get('schema') != VERSION or _digest(value) != digest or value['head'].get('rules') != scoped.rule_manifest():
        raise ValueError('Temporal model binding differs')
    return value
