"""Observable prefix metadata; no labels, gains, or inferred semantic facts."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
import math

METADATA_NAMES = ['log1p_days_before_match','publication_time_unknown','historical_version_unknown',
    'source_is_official','log1p_body_characters','historical_reconstruction','exact_prior_body_duplicate',
    'candidate_claims_unknown','adopted_claims_unknown','repeated_claim_count','existing_multistatus_count',
    'candidate_introduced_multistatus_count','candidate_expanded_multistatus_count','factual_update_unknown','source_official_unknown',
    'exact_home_team_mentions','exact_away_team_mentions','target_aliases_unknown','target_aliases_partial',
    'source_target_home','source_target_away','source_target_unknown']


def _date(s):
    try:return datetime.fromisoformat(s.replace('Z','+00:00')).date()
    except (AttributeError,TypeError,ValueError):return None


def _claims(document):
    claims=document.get('existing_claims')
    if not isinstance(claims,list):return None
    values=set()
    for c in claims:
        side=c.get('team_side') or c.get('team') or c.get('mapped_team')
        entity=c.get('entity_normalized') or c.get('entity_name') or c.get('entity') or c.get('mapped_entity')
        kind=c.get('kind') or c.get('mapped_kind');value=c.get('value') or c.get('mapped_value')
        if not all(isinstance(x,str) and x for x in [side,entity,kind,value]):continue
        values.add((side,' '.join(entity.casefold().split()),c.get('scope') or 'UNKNOWN',kind,value))
    return values


def _status(claims):
    result=defaultdict(set)
    for team,entity,scope,kind,value in claims:
        if kind=='availability':result[(team,entity,scope)].add(value)
    return result


def document_metadata(document, adopted_documents, match_date):
    """Recompute for actual adopted prefix. Unknown counts are 0 with flags=1.

    Use link-specific published_at/source_type where a body hash has multiple
    source records. Multistatus is not a truth or contradiction judgment.
    factual_update_unknown=1: publication ordering never proves a fact update.
    """
    candidate=_claims(document);known=[_claims(d) for d in adopted_documents]
    previous=set().union(*(v for v in known if v is not None)) if known else set()
    cv=candidate or set();prior=_status(previous);after=_status(previous|cv)
    existing=sum(len(v)>1 for v in prior.values())
    introduced=sum(len(v)>1 and len(prior.get(k,set()))<=1 for k,v in after.items())
    expanded=sum(len(prior.get(k,set()))>1 and len(v)>len(prior[k]) for k,v in after.items())
    pub=_date(document.get('published_at') or document.get('source_published_at'));target=_date(match_date)
    elapsed=(target-pub).days if target and pub else None
    st=document.get('source_type');official=st in ('official_club','official_league','official')
    source_unknown=not isinstance(st,str) or st in ('unknown','UNKNOWN')
    text_sha=document.get('text_sha256') or document.get('body_sha256')
    duplicate=bool(text_sha) and any((d.get('text_sha256') or d.get('body_sha256'))==text_sha for d in adopted_documents)
    result=[math.log1p(max(0,elapsed)) if elapsed is not None else 0.,float(elapsed is None),
        float(document.get('historical_version_unknown',True)),float(official),math.log1p(document.get('body_characters',0)),1.,
        float(duplicate),float(candidate is None),float(any(v is None for v in known)),float(len(cv&previous)),float(existing),
        float(introduced),float(expanded),1.,float(source_unknown)]
    binding=document.get('target_binding') or {}
    result.extend([float(binding.get('exact_home_team_mentions',0)),float(binding.get('exact_away_team_mentions',0)),
        float(binding.get('target_aliases_unknown',True)),float(binding.get('target_aliases_partial',True)),
        float(binding.get('source_target_home',False)),float(binding.get('source_target_away',False)),
        float(binding.get('source_target_unknown',True))])
    return result


def sequence_metadata(documents, match_date):
    return [document_metadata(d,documents[:i],match_date) for i,d in enumerate(documents)]
