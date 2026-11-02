import math

from football_research.acquisition.full_history_sequence_metadata import (
    METADATA_NAMES, document_metadata,
)


def claim(value):
    return {'team_side':'home','entity_normalized':'named player','scope':'this_match',
            'kind':'availability','value':value}


def doc(value=None, key='a'):
    return {'text_sha256':key,'published_at':'2020-01-01T12:00:00Z',
            'body_characters':1000,'source_type':'official_club',
            'historical_version_unknown':True,'existing_claims':value}


def meta(d, adopted=()):
    return dict(zip(METADATA_NAMES,document_metadata(d,list(adopted),'2020-01-03')))


def test_unknown_claims_are_not_confirmed_empty_claims():
    unknown=meta(doc());empty=meta(doc([]))
    assert unknown['candidate_claims_unknown']==1
    assert empty['candidate_claims_unknown']==0
    assert unknown['repeated_claim_count']==empty['repeated_claim_count']==0


def test_relations_recompute_for_actual_prefix():
    absent=doc([claim('unavailable')],'absent')
    available=doc([claim('available')],'available')
    assert meta(available)['candidate_introduced_multistatus_count']==0
    assert meta(available,[absent])['candidate_introduced_multistatus_count']==1
    reused=meta(absent,[absent,available])
    assert reused['existing_multistatus_count']==1
    assert reused['candidate_introduced_multistatus_count']==0
    assert reused['repeated_claim_count']==1
    assert reused['exact_prior_body_duplicate']==1


def test_unknown_prefix_is_flagged_without_fabricated_claims():
    result=meta(doc([claim('unavailable')]),[doc(None,'other')])
    assert result['adopted_claims_unknown']==1
    assert result['repeated_claim_count']==0
    assert result['factual_update_unknown']==1


def test_publication_time_does_not_establish_factual_update():
    d=doc([claim('available')]);d['published_at']='2020-01-02T12:00:00Z'
    result=meta(d,[doc([claim('unavailable')],'earlier')])
    assert result['factual_update_unknown']==1
    assert result['candidate_introduced_multistatus_count']==1


def test_future_targets_do_not_change_observable_metadata():
    d=doc([claim('unavailable')]);same=dict(d,realized_gain=100,p_after=[1,0,0],outcome_class=2)
    assert meta(d)==meta(same)


def test_missing_source_and_clock_have_explicit_flags():
    d=doc([]);d['source_type']=None;d['published_at']=None
    result=meta(d)
    assert result['publication_time_unknown']==1
    assert result['source_official_unknown']==1
    assert result['source_is_official']==0
    assert len(result)==22 and all(math.isfinite(v) for v in result.values())


def test_nonavailability_predicates_are_not_counted_as_status_conflict():
    a=claim('starter');a['kind']='selection'
    b=claim('bench');b['kind']='selection'
    assert meta(doc([b]),[doc([a])])['candidate_introduced_multistatus_count']==0


def test_unobserved_target_binding_is_unknown_not_false_fact():
    result=meta(doc([]))
    assert result['target_aliases_unknown']==1
    assert result['source_target_unknown']==1
    assert result['exact_home_team_mentions']==0


def test_observed_target_binding_reaches_model_metadata():
    d=doc([]);d['target_binding']={'exact_home_team_mentions':4,'exact_away_team_mentions':2,
        'target_aliases_unknown':False,'target_aliases_partial':True,
        'source_target_home':False,'source_target_away':True,'source_target_unknown':False}
    result=meta(d)
    assert result['exact_home_team_mentions']==4 and result['exact_away_team_mentions']==2
    assert result['source_target_away']==1 and result['source_target_unknown']==0


def test_lexical_team_match_uses_boundaries_and_deduplicates_alias_spans():
    from tools.augment_full_history_target_binding import exact_mentions
    assert exact_mentions('ArsenalW Arsenal Women Arsenal.', ['Arsenal'])==2
    assert exact_mentions('Manchester United Manchester\nUnited', ['Manchester United','United'])==2
