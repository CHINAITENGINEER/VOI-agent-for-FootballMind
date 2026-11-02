"""Round155: local archive pages can prove coverage without hidden enrichment."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from football_agent.experience import public_state, validate_decision
from football_agent.local_evidence import capability, project_query
from football_agent.acquisition_replay import verify_run
from football_agent.runtime_fixture_bridge import prepare_fixture_identity, validate_fixture_identity
from football_agent.runtime_identity_pages import local_identity_candidate

H = runpy.run_path(str(Path(__file__).with_name('test_runtime_fixture_bridge.py')))
fixture_row, recent_payload, setup = H['fixture_row'], H['recent_payload'], H['setup']
LOCAL = runpy.run_path(str(Path(__file__).with_name('test_local_evidence.py')))


def _public_page(page):
    result = {key: deepcopy(page.get(key)) for key in (
        'document_id', 'url', 'title', 'status', 'content_kind', 'signals', 'observed_at',
        'source_published_at', 'raw_sha256', 'request_id', 'semantic_relevance',
        'acquisition_id', 'selection_operation_id')}
    lines = page.get('text', '').splitlines()
    result.update(text_truncated=page.get('text_truncated', False),
                  excerpt='\n'.join(lines), excerpt_truncated=page.get('text_truncated', False),
                  links=deepcopy(page.get('links', [])),
                  local_evidence=deepcopy(page.get('local_evidence')),
                  structured_records=[json.loads(line) for line in lines])
    return result


def _bound_pages(tmp_path, offsets):
    rows = [fixture_row(matchId=1000 + i, home='构造甲队' + str(i)) for i in range(80)]
    rows[-1]['home'] = '窗口外目标甲队'
    archive, public, originals = setup(
        tmp_path, rows, match='2026-10-01 窗口外目标甲队 vs 构造乙队')
    parent = originals[0]
    requested = list(offsets)
    if len(requested) > 1:
        requested = []
        offset = 0
        while offset is not None:
            requested.append(offset)
            probe = project_query(archive, [parent], {
                'document_id': parent['document_id'], 'record_kind': 'fixture',
                'source_match_id': None, 'team_name': None, 'team_match': 'exact',
                'offset': offset, 'limit': 20}, 'probe-' + str(offset))
            offset = probe['local_evidence']['next_offset']
    for offset in requested:
        page = project_query(archive, [parent], {
            'document_id': parent['document_id'], 'record_kind': 'fixture',
            'source_match_id': None, 'team_name': None, 'team_match': 'exact',
            'offset': offset, 'limit': 20}, 'round155-page-' + str(offset))
        page['acquisition_id'] = 'local-acq-' + str(offset)
        page['selection_operation_id'] = 'selection-' + str(offset)
        originals.append(page)
        public['documents'].append(_public_page(page))
    return archive, public, originals


def test_complete_gap_free_local_pages_replace_truncated_parent_for_identity(tmp_path):
    archive, public, originals = _bound_pages(tmp_path, [0, 20, 40, 60])
    wrapper = prepare_fixture_identity(archive, public, originals, include_local=True)
    assert wrapper['version'] == 'runtime-fixture-bridge-v2'
    assert not wrapper['errors']
    assert len(wrapper['local_coverage']) == 1
    assert wrapper['local_coverage'][0]['complete'] is True
    assert wrapper['local_coverage'][0]['covers_input'] is True
    assert wrapper['resolution']['status'] == 'resolved'
    assert wrapper['resolution']['target_key'] == 'spftll:1079'
    public['fixture_identity'] = wrapper
    assert validate_fixture_identity(public) == wrapper['resolution']


def test_partial_local_page_keeps_identity_incomplete(tmp_path):
    archive, public, originals = _bound_pages(tmp_path, [0])
    wrapper = prepare_fixture_identity(archive, public, originals, include_local=True)
    assert not wrapper['errors']
    assert wrapper['local_coverage'][0]['complete'] is False
    assert wrapper['local_coverage'][0]['next_offset'] is not None
    assert wrapper['resolution']['status'] == 'incomplete'
    assert wrapper['resolution']['target_key'] is None


def test_local_identity_selector_requires_coverage_capable_normalized_query():
    parent = {'document_id': 'doc-parent', 'content_kind': 'platform_structured',
              'status': 'readable', 'platform': {'source': 'spftll', 'quality': 'normalized', 'issues': 0}}
    public = {
        'fixture_identity': {'errors': [], 'verified_sources': [{'document_id': 'doc-parent'}],
                             'verified_local_sources': [], 'local_coverage': []},
        'local_query_capability': {'version': 'acquired-platform-query-v2', 'action': 'query_acquired',
                                   'max_reads': 8, 'window_chars': 4500,
                                   'scope': 'already_acquired_platform_documents_in_this_task_only',
                                   'matching': 'literal_or_nfkc_casefold_whitespace_exact_no_fuzzy_alias_or_ranking',
                                   'ordering': 'source_order_after_explicit_filters', 'http_requests': 0,
                                   'latency_seconds_estimate': None, 'decision_loss_equivalent': None,
                                   'numeric_voi_calibrated': False},
        'documents': [parent], 'remaining': {'local_reads': 2},
        'final_decision_only': False, 'seconds_to_deadline': 30,
    }
    identity = {'status': 'incomplete', 'input_constraints': {
        'kind': 'team_date', 'date': '2026-10-01', 'teams': ['窗口外目标甲队', '构造乙队']}}
    def candidate(name, mode, cid):
        return SimpleNamespace(candidate_id=cid, response={'action': 'query_acquired', 'local_query': {
            'document_id': 'doc-parent', 'record_kind': 'fixture', 'source_match_id': None,
            'team_name': name, 'team_match': mode, 'offset': 0, 'limit': 20}})
    batch = SimpleNamespace(candidates=[candidate('窗口外目标甲队', 'exact', 'unsafe'),
                                        candidate('窗口外目标甲队', 'normalized_exact', 'safe')])
    assert local_identity_candidate(public, batch, identity).candidate_id == 'safe'


def test_v1_snapshot_validation_preserves_literal_matching(tmp_path):
    service, run, _, _ = LOCAL['fixture_runtime'](tmp_path)
    try:
        service.step(run)
        public = public_state(service.step(run))
        public['local_query_capability'] = capability(legacy=True)
        response = LOCAL['local_decision'](public, team_name='窗口外目标甲队')
        validate_decision(response, public)
        response['local_query']['team_match'] = 'normalized_exact'
        with pytest.raises(ValueError, match='LOCAL_QUERY_LEGACY_MATCHING'):
            validate_decision(response, public)
        public['local_query_capability'] = capability()
        validate_decision(response, public)
    finally:
        service.close()


def test_archived_v1_local_run_replays_with_current_validator(tmp_path, monkeypatch):
    # Generate a durable run with the frozen v1 wire contract, then restore the
    # current runtime before replay. No snapshots or receipts are rewritten.
    with monkeypatch.context() as legacy:
        legacy.setattr('football_agent.experience.local_capability',
                       lambda **kwargs: capability(legacy=True))
        legacy.setattr('football_agent.experience.project_query',
                       lambda *args: project_query(*args, legacy=True))
        service, run, _, _ = LOCAL['fixture_runtime'](tmp_path)
        try:
            service._drive(run, True)
            folder = service.folder(run)
            state = service.get(run)
            assert state['config']['local_query_capability'] == capability(legacy=True)
            assert state['documents'][-1]['local_evidence']['version'] == capability(legacy=True)['version']
        finally:
            service.close()
    replay = verify_run(folder)
    assert replay['accepted_decisions_verified'] == 3
    assert replay['linked_local_reads'] == replay['local_documents_reconstructed'] == 1
    assert replay['real_calls_made_by_replay'] == 0


@pytest.mark.parametrize('truncated', [False, True])
def test_filtered_coverage_preserves_visible_cross_document_id_conflict(tmp_path, truncated):
    archive, public, originals = setup(tmp_path, [fixture_row()])
    rows = [fixture_row(home='另一支甲队', away='另一支乙队'),
            fixture_row(matchId=2002, away='另外对手')]
    if truncated:
        rows += [fixture_row(matchId=3000+i, home='其他队'+str(i)) for i in range(80)]
    _, second, more = setup(tmp_path, rows)
    public['snapshot_as_of'] = second['snapshot_as_of']
    public['documents'].extend(second['documents'])
    originals.extend(more)
    before = prepare_fixture_identity(archive, public, originals, include_local=True)
    assert before['resolution']['status'] == 'conflicting'
    parent = more[0]
    page = project_query(archive, [parent], {
        'document_id': parent['document_id'], 'record_kind': 'fixture',
        'source_match_id': None, 'team_name': '构造甲队', 'team_match': 'normalized_exact',
        'offset': 0, 'limit': 20}, 'round155-conflict')
    originals.append(page)
    public['documents'].append(_public_page(page))
    wrapper = prepare_fixture_identity(archive, public, originals, include_local=True)
    assert not wrapper['errors']
    assert wrapper['local_coverage'][0]['usable_for_identity'] is True
    assert wrapper['resolution']['status'] == 'conflicting'
    assert wrapper['resolution']['target_key'] is None
    public['fixture_identity'] = wrapper
    assert validate_fixture_identity(public) == wrapper['resolution']


def test_complete_team_query_keeps_conflict_from_other_verified_local_query(tmp_path):
    archive, public, originals = setup(tmp_path, [fixture_row()])
    rows = [fixture_row(matchId=2002, away='另外对手')]
    rows += [fixture_row(matchId=3000+i, home='其他队'+str(i)) for i in range(80)]
    rows += [fixture_row(home='另一支甲队', away='另一支乙队')]
    _, second, more = setup(tmp_path, rows)
    public['snapshot_as_of'] = second['snapshot_as_of']
    public['documents'].extend(second['documents'])
    originals.extend(more)
    parent = more[0]
    assert all(row['source_id'] != '1001' for row in second['documents'][0]['structured_records'])
    for mode, filters in [('id', {'source_match_id': '1001', 'team_name': None}),
                          ('team', {'source_match_id': None, 'team_name': '构造甲队'})]:
        page = project_query(archive, [parent], {
            'document_id': parent['document_id'], 'record_kind': 'fixture',
            'team_match': 'normalized_exact', 'offset': 0, 'limit': 20,
            **filters}, 'round155-conflict-' + mode)
        originals.append(page)
        public['documents'].append(_public_page(page))
    wrapper = prepare_fixture_identity(archive, public, originals, include_local=True)
    assert not wrapper['errors']
    assert sum(group['usable_for_identity'] for group in wrapper['local_coverage']) == 1
    assert wrapper['resolution']['status'] == 'conflicting'
    assert wrapper['resolution']['target_key'] is None
    public['fixture_identity'] = wrapper
    assert validate_fixture_identity(public) == wrapper['resolution']
