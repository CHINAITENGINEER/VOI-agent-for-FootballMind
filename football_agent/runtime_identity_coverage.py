"""Prove that visible local pages cover the user's possible fixture matches.

No archive reads and no hidden records here. Page counts and offsets originate
from the replayed local tool. Only a complete, gap-free query whose predicate
includes every possible name match can replace its truncated parent projection.
"""
from copy import deepcopy

from .audit import stored_digest as digest
from .local_evidence import normalize_team_name
from .runtime_fixture_identity import resolve_match_identity

VERSION = 'runtime-identity-coverage-v1'


def query_covers_input(query, constraints):
    if query.get('record_kind') not in {'fixture', 'all'} or query.get('source_match_id') is not None:
        return False
    if query.get('team_name') is None:
        return True
    # Literal/case-sensitive filters can omit NFKC/case variants accepted by
    # the identity resolver. They remain useful pages but not coverage proofs.
    return (constraints.get('kind') == 'team_date'
            and query.get('record_kind') == 'fixture'
            and query.get('team_match') == 'normalized_exact'
            and normalize_team_name(query['team_name']) in constraints.get('teams', []))


def coverage_views(public, receipts):
    indexed = {doc['document_id']: doc for doc in public['documents']}
    constraints = resolve_match_identity({**public, 'documents': []})['input_constraints']
    groups = {}
    for receipt in receipts:
        doc = indexed[receipt['document_id']]
        query = doc['local_evidence']['query']
        key = digest({k: v for k, v in query.items() if k not in {'offset', 'limit'}})
        groups.setdefault(key, []).append(doc)
    coverage, derived, replaced = [], [], set()
    for key, pages in sorted(groups.items()):
        pages = sorted(pages, key=lambda doc: (doc['local_evidence']['offset'], doc['document_id']))
        first = pages[0]['local_evidence']
        parent = indexed[first['source_document_id']]
        meta = parent['platform']
        applicable = query_covers_input(first['query'], constraints)
        normalized = (meta.get('quality') in {'normalized', 'no_data', 'projection_window_exceeded'}
                      and meta.get('issues') == 0)
        offset, total, valid, ended = 0, first['records_matched'], True, False
        for page in pages:
            local = page['local_evidence']
            count = len(page.get('structured_records', []))
            valid = valid and (not ended and local['offset'] == offset
                and local['records_matched'] == total and local['records_returned'] == count
                and local['omitted_oversized_offsets'] == []
                and local['source_quality'] == meta['quality']
                and (page['status'] == 'readable' or count == total == 0))
            offset += count
            ended = local['next_offset'] is None
            valid = valid and (offset == total if ended else local['next_offset'] == offset and count > 0)
        complete = bool(valid and ended and offset == total)
        usable = complete and applicable and normalized
        coverage.append({'query_digest': key, 'source_document_id': parent['document_id'],
            'document_ids': [page['document_id'] for page in pages],
            'records_matched': total, 'visible_records': sum(len(page['structured_records']) for page in pages),
            'complete': complete, 'covers_input': applicable, 'source_normalized': normalized,
            'usable_for_identity': usable,
            'next_offset': None if complete else pages[-1]['local_evidence']['next_offset']})
        if not usable:
            continue
        replaced.add(parent['document_id'])
    # Every verified visible page remains evidence, even when its query does
    # not cover all possible matches. Otherwise a complete narrow query could
    # erase a conflicting row already read through a different local query.
    # Only `replaced` discharges a parent's coverage gap; partial pages alone
    # never grant completeness.
    for receipt in receipts:
        page = indexed[receipt['document_id']]
        parent = indexed[page['local_evidence']['source_document_id']]
        meta = parent['platform']
        if meta.get('quality') not in {'normalized', 'no_data', 'projection_window_exceeded'} or meta.get('issues') != 0:
            continue
        doc = deepcopy(page)
        doc.update(content_kind='platform_structured', status='readable',
                   text_truncated=False, excerpt_truncated=False)
        doc['platform'] = deepcopy(meta)
        doc['platform'].update(quality='normalized', records_visible=len(doc['structured_records']))
        derived.append(doc)
    return coverage, derived, replaced


__all__ = ['VERSION', 'query_covers_input', 'coverage_views']
