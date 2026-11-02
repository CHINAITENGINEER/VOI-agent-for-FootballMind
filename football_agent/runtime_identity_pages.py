"""Deterministic local-page prerequisite selection for fixture identity.

This is a prerequisite route, not a calibrated VoI estimate. It lets the
state-bound selector spend a local read when the user supplied a team/date
match but the bounded catalogue projection is incomplete. The candidate still
has to be proposed by the Agent and pass the normal decision validator.
"""
from .local_evidence import ACTION as LOCAL_QUERY, capability as local_capability
from .runtime_identity_coverage import query_covers_input
from .audit import stored_digest as digest

VERSION = 'runtime-identity-pages-v1'


def _query_candidate(candidate, public, identity):
    response = candidate.response
    if response.get('action') != LOCAL_QUERY:
        return False
    query = response.get('local_query')
    if not isinstance(query, dict):
        return False
    if not query_covers_input(query, identity['input_constraints']):
        return False
    parent = next((document for document in public['documents']
                   if document.get('document_id') == query.get('document_id')), None)
    if not isinstance(parent, dict) or parent.get('content_kind') != 'platform_structured':
        return False
    platform = parent.get('platform') or {}
    verified = {entry['document_id'] for entry in public['fixture_identity']['verified_sources']}
    if (platform.get('source') != 'spftll' or parent['document_id'] not in verified
            or platform.get('quality') not in {'normalized', 'no_data', 'projection_window_exceeded'}
            or platform.get('issues') != 0):
        return False
    key = digest({k: v for k, v in query.items() if k not in {'offset', 'limit'}})
    prior = next((group for group in public['fixture_identity'].get('local_coverage', [])
                  if group['query_digest'] == key), None)
    if prior:
        return not prior['complete'] and prior['next_offset'] is not None and query['offset'] == prior['next_offset']
    return query['offset'] == 0


def local_identity_candidate(public, batch, identity):
    """Choose one proposed exact team page while identity is unresolved.

    Selection is stable by query document, offset, literal name and candidate
    ID. The LLM's recommendation is never treated as an information value.
    """
    if not isinstance(identity, dict) or identity.get('status') not in {'unresolved', 'incomplete'}:
        return None
    wrapper = public.get('fixture_identity')
    if not isinstance(wrapper, dict) or wrapper.get('errors') or 'verified_local_sources' not in wrapper:
        return None
    constraints = identity.get('input_constraints') or {}
    if constraints.get('kind') != 'team_date' or not constraints.get('date'):
        return None
    if public.get('final_decision_only') or public.get('seconds_to_deadline', 1) <= 0:
        return None
    capability = public.get('local_query_capability')
    if capability != local_capability() or public.get('remaining', {}).get('local_reads', 0) <= 0:
        return None
    documents = public.get('documents')
    if not isinstance(documents, list):
        return None
    proposed = [candidate for candidate in batch.candidates if _query_candidate(candidate, public, identity)]
    return min(proposed, key=lambda candidate: (
        candidate.response['local_query']['document_id'],
        candidate.response['local_query']['offset'],
        candidate.response['local_query']['team_name'] or '',
        candidate.candidate_id)) if proposed else None


__all__ = ['VERSION', 'local_identity_candidate']
