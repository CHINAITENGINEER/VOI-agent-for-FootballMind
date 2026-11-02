"""Bounded identity prerequisite, explicitly outside empirical VoI claims."""
from .structured_sources import registered_request, source_catalog

VERSION = 'runtime-identity-discovery-v1'
MAX_SOURCE_ACTIONS = 1


def is_identity_source(response):
    if response.get('action') != 'read_url':
        return False
    spec = registered_request(response.get('url') or '')
    return spec is not None and spec['kind'] in {'recent', 'lottery'}


def identity_source_attempts(responses):
    # Accepted attempts consume the cap even if transport fails or returns no document.
    return sum(is_identity_source(response) for response in responses)


def discovery_candidate(public, batch, identity):
    """Select only a proposed, registered catalogue action, never invent a call.

    This is a deterministic prerequisite allowance, not a learned value ranking.
    Missing counters, failed evidence and partial coverage do not reopen it.
    """
    attempts = public.get('identity_source_attempts')
    if type(attempts) is not int or attempts < 0 or attempts >= MAX_SOURCE_ACTIONS:
        return None
    if (identity.get('status') != 'unresolved' or identity.get('issues')
            or (identity.get('input_constraints') or {}).get('kind') != 'team_date'
            or not identity['input_constraints'].get('date')):
        return None
    if (public.get('final_decision_only') or public.get('remaining', {}).get('http_calls', 0) <= 0
            or public.get('seconds_to_deadline', 1) <= 0):
        return None
    # The current capability catalogue is part of the runtime snapshot.
    if public.get('source_catalog') != source_catalog():
        return None
    urls = {entry['url'] for entry in public['source_catalog']['entries']}
    proposed = [candidate for candidate in batch.candidates if
                is_identity_source(candidate.response) and candidate.response['url'] in urls]
    # Stable URL tie-break avoids borrowing the LLM's recommendation as utility.
    return min(proposed, key=lambda c: (c.response['url'], c.candidate_id)) if proposed else None


__all__ = ['VERSION', 'MAX_SOURCE_ACTIONS', 'identity_source_attempts', 'discovery_candidate']
