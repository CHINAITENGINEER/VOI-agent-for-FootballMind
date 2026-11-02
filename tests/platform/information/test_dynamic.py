"""Round-four dynamic quality/dependency/time contract tests.

The export is an existing immutable COMPLETE fixture; all changes here are
in-memory synthetic mechanisms and never write the production archive.
"""
from datetime import timedelta
from pathlib import Path
import json

import pytest

from football_platform.information import DynamicEpisodeEnvironment, SCENARIOS

BUNDLE = Path('var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda')


def make(scenario):
    return DynamicEpisodeEnvironment.from_export(BUNDLE, scenario=scenario, seed=17)


def test_catalog_exposes_dependency_refresh_and_cost_range_without_values():
    env = make('complementary')
    items = env.catalog()
    assert all(item.dependency_key and item.refresh_window and item.cost_range for item in items)
    assert items[0].interaction_key == items[1].interaction_key
    encoded = json.dumps([item.model_dump(mode='json') for item in items])
    assert 'home_elo' not in encoded and 'target' not in encoded


def test_stale_episode_changes_mask_and_freshness():
    env = make('stale')
    result = env.acquire('stale.strength', 'stale-1')
    assert result.status == 'ACQUIRED'
    group = env.visible_state().groups['strength']
    assert group.expires_at is not None and group.freshness_age >= 0
    state = env.advance_time(group.expires_at + timedelta(seconds=1))
    assert state.mask['strength'] == 'STALE'
    assert state.groups['strength'].quality_status == 'stale'
    assert state.groups['strength'].quality == 'STALE'


def test_duplicate_is_not_new_information_and_is_charged_as_simulated_refresh():
    env = make('duplicate')
    first = env.acquire('primary.strength', 'a')
    second = env.acquire('duplicate.strength', 'b')
    assert first.status == 'ACQUIRED'
    assert second.status == 'UNCHANGED'
    assert second.charged_cost.requests.amount == 1
    assert second.revealed_groups['strength'].quality_status == 'duplicate'
    assert env.visible_state().groups['strength'].quality_status == 'valid'


def test_no_change_is_distinct_from_duplicate():
    env = make('no_change')
    env.acquire('primary.strength', 'a')
    result = env.acquire('refresh.no_change', 'b')
    assert result.status == 'UNCHANGED'
    assert result.revealed_groups['strength'].quality_status == 'no_change'


def test_conflicting_sources_are_visible_but_not_model_observed():
    env = make('conflict')
    assert env.acquire('provider_a.strength', 'a').status == 'ACQUIRED'
    result = env.acquire('provider_b.strength', 'b')
    state = env.visible_state()
    assert result.status == 'FAILED'
    assert result.revealed_groups['strength'].quality_status == 'conflict'
    assert state.mask['strength'] == 'FAILED'
    assert state.groups['strength'].quality_status == 'conflict'
    assert state.groups['strength'].source_id.count('|conflict|') == 1


def test_no_data_does_not_remove_second_legal_action():
    env = make('no_data_then_available')
    first = env.acquire('first.no_data', 'a')
    assert first.status == 'NO_DATA'
    assert env.visible_state().mask['strength'] == 'SOURCE_MISSING'
    assert env.acquire('retry.strength', 'b').status == 'ACQUIRED'
    assert env.visible_state().mask['strength'] == 'OBSERVED'


def test_provider_error_is_separate_and_retryable():
    env = make('provider_error')
    first = env.acquire('provider.error', 'a')
    assert first.status == 'FAILED'
    assert first.revealed_groups['strength'].quality_status == 'provider_error'
    assert env.acquire('provider.retry', 'b').status == 'ACQUIRED'


def test_complementary_actions_remain_individually_available_and_jointly_revealed():
    env = make('complementary')
    catalog = env.catalog()
    assert all(item.capability_status == 'AVAILABLE' for item in catalog)
    one = env.acquire('complementary.strength', 'a')
    assert one.status == 'ACQUIRED'
    assert env.visible_state().mask['recent_form'] == 'UNACQUIRED'
    two = env.acquire('complementary.recent_form', 'b')
    assert two.status == 'ACQUIRED'
    assert env.visible_state().mask['recent_form'] == 'OBSERVED'


def test_shared_response_registry_distinguishes_query_cache_view_and_refresh():
    env = make('shared_response')
    specs = {item.action_id: item for item in env.action_registry()}
    assert [item.action_id for item in env.action_registry()] == sorted(specs)
    assert specs['shared.query.strength'].operation == 'query'
    assert specs['shared.cache.recent_form'].cache_policy == 'cache_view'
    assert specs['shared.cache.recent_form'].expected_cost.requests.amount == 0
    assert specs['shared.refresh.rolling_xg'].operation == 'refresh'
    assert specs['shared.query.strength'].dependency_key == specs['shared.cache.recent_form'].dependency_key
    assert specs['shared.query.strength'].interaction_key == specs['shared.refresh.rolling_xg'].interaction_key

    first = env.acquire('shared.query.strength', 'shared-1')
    assert first.status == 'ACQUIRED' and first.charged_cost.requests.amount == 1
    state = env.visible_state()
    assert state.mask['strength'] == 'OBSERVED'
    assert state.mask['recent_form'] == 'UNACQUIRED'
    assert state.mask['rolling_xg'] == 'UNACQUIRED'
    cache = env.acquire('shared.cache.recent_form', 'shared-2')
    assert cache.status == 'CACHE_HIT' and cache.charged_cost.requests.amount == 0
    assert cache.revealed_groups['recent_form'].dependency_key == specs['shared.query.strength'].dependency_key
    refreshed = env.acquire('shared.refresh.rolling_xg', 'shared-3')
    assert refreshed.status == 'ACQUIRED' and refreshed.charged_cost.requests.amount == 1
    assert env.visible_state().mask['rolling_xg'] == 'OBSERVED'


def test_refresh_cycle_replaces_stale_quality_without_backdating_cutoff():
    env = make('refresh_cycle')
    first = env.acquire('refresh.query.strength', 'refresh-1')
    assert first.status == 'ACQUIRED'
    old = env.visible_state().groups['strength']
    env.advance_time(old.expires_at + timedelta(seconds=1))
    assert env.visible_state().mask['strength'] == 'STALE'
    result = env.acquire('refresh.refresh.strength', 'refresh-2')
    assert result.status == 'ACQUIRED'
    current = env.visible_state().groups['strength']
    assert current.quality_status == 'valid' and current.quality == 'VALIDATED'
    assert current.expires_at > current.available_at
    assert current.available_at == env.visible_state().groups['strength'].available_at


def test_action_registry_and_trace_are_deterministic_for_new_scenarios():
    for scenario, actions in {
        'shared_response': [('shared.query.strength', 'a'), ('shared.cache.recent_form', 'b'),
                            ('shared.refresh.rolling_xg', 'c')],
        'refresh_cycle': [('refresh.query.strength', 'a'), ('refresh.refresh.strength', 'b')],
    }.items():
        left, right = make(scenario), make(scenario)
        assert left.configuration()['action_registry_hash'] == right.configuration()['action_registry_hash']
        assert [x.model_dump(mode='json') for x in left.action_registry()] == [x.model_dump(mode='json') for x in right.action_registry()]
        for action_id, request_id in actions:
            assert left.acquire(action_id, request_id).model_dump(mode='json') == right.acquire(action_id, request_id).model_dump(mode='json')
        assert left.visible_state().model_dump(mode='json') == right.visible_state().model_dump(mode='json')


def test_same_request_is_idempotent_and_does_not_recharge():
    env = make('fresh')
    first = env.acquire('fresh.strength', 'same')
    repeat = env.acquire('fresh.strength', 'same')
    assert first.status == repeat.status == 'ACQUIRED'
    assert repeat.repeated_request and repeat.charged_cost.requests.amount == 0
    assert env.visible_state().budget.spent['requests'] == 1


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_dynamic_public_state_has_no_archive_path_or_label(scenario):
    env = make(scenario)
    state = env.visible_state().model_dump_json()
    assert 'understat-2025-historical_reconstruction-a3b3d45811fd5cda' not in state
    assert 'target_label' not in state and 'full_snapshot' not in state


def test_refresh_time_is_consistent_in_both_consumer_projections():
    from football_research.acquisition.platform_bridge import to_research_state
    from football_agent.research_bridge import platform_state
    env = make('refresh_cycle')
    env.acquire('refresh.query.strength', 'before')
    at = env.visible_state().groups['strength'].expires_at + timedelta(seconds=1)
    env.advance_time(at)
    env.acquire('refresh.refresh.strength', 'after')
    state = env.visible_state()
    assert state.prediction_cutoff == at
    assert state.groups['strength'].times.feature_available_at == at
    assert state.groups['strength'].available_at <= state.prediction_cutoff
    assert to_research_state(state)['groups']['elo']['status'] == 'observed'
    assert platform_state(state)['mask']['elo']


def test_simulated_budget_is_enforced_before_new_evidence_or_charge():
    env = make('fresh')
    for i in range(4):
        env.acquire('fresh.strength', f'request-{i}')
    before = env.visible_state()
    blocked = env.acquire('fresh.strength', 'over-budget')
    assert blocked.status == 'BUDGET_EXCEEDED'
    assert blocked.charged_cost.requests.amount == 0
    assert before == env.visible_state()


def test_cache_read_without_query_does_not_reveal_any_hidden_values():
    env = make('shared_response')
    result = env.acquire('shared.cache.recent_form', 'empty-cache')
    assert result.status == 'NO_DATA'
    assert env.visible_state().mask['recent_form'] == 'SOURCE_MISSING'
    assert 'recent_form' not in env.visible_state().groups
    assert not result.revealed_groups['recent_form'].values
    assert env.visible_state().budget.spent['requests'] == 0
