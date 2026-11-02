from football_spider.models.contracts import EntityRef, MatchRecord, NormalizedBatch, Score
from football_spider.quality.entity_mapping import entity_mapping_state


def match(home_quality="stable", away_quality="stable", competition_quality="stable"):
    return MatchRecord(
        source_id="m-1",
        competition=EntityRef(source_id="c-1", name="League", id_quality=competition_quality),
        home=EntityRef(source_id="h-1", name="Home", id_quality=home_quality),
        away=EntityRef(source_id="a-1", name="Away", id_quality=away_quality),
        kickoff="2026-09-28T12:00:00Z", status="scheduled",
        original_time="2026-09-28T12:00:00Z", source_timezone="UTC",
        score=Score(), half_score=Score(),
    )


def test_source_level_mapping_state_is_derived_from_normalized_entities():
    assert entity_mapping_state(NormalizedBatch(matches=[match()])) == "mapped"
    assert entity_mapping_state(NormalizedBatch(matches=[match(home_quality="source_name")])) == "ambiguous"
    assert entity_mapping_state(NormalizedBatch(matches=[match(home_quality="match_scoped")])) == "unmapped"
    assert entity_mapping_state(NormalizedBatch()) == "not_evaluated"
