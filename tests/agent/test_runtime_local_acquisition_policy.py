"""Local evidence is an optional, costed information action after identity."""
from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from football_agent import runtime_discovered_routes as discovery, runtime_exploration as exploration
from football_agent import runtime_local_acquisition as local, runtime_source_claims as claims
from football_agent.acquisition_boundary import policy_identity
from football_agent.local_evidence import project_query
from football_agent.runtime_fixture_bridge import prepare_fixture_identity

D = runpy.run_path(str(Path(__file__).with_name('test_runtime_discovered_routes_policy.py')))
F = runpy.run_path(str(Path(__file__).with_name('test_runtime_fixture_bridge.py')))
L = runpy.run_path(str(Path(__file__).with_name('test_round155_local_identity.py')))
PROFILE = {'search_web': {'bing': .1}, 'read_url': .1, 'query_acquired': .05}


def setup(root, *, profile=None, count=3, max_cost=.3, rows=12):
    selector = D['policy'](profile=deepcopy(PROFILE) if profile is None else profile,
                           count=count, cost=max_cost)
    fixtures = [F['fixture_row'](matchId=1001+i, home='构造甲队' + (str(i) if i else ''),
        matchTime='2027-10-01 20:00:00', spfWinFoot=2.1, spfEqualFoot=3.2, spfLoseFoot=3.5,
        goalFoot=-1, rspfWinFoot=3.0, rspfEqualFoot=3.4, rspfLoseFoot=2.0) for i in range(rows)]
    archive, public, originals = F['setup'](root, {'code':0,'data':[{'list':fixtures}]},
        kind='lottery', match='2027-10-01 构造甲队 vs 构造乙队')
    public[claims.CAPABILITY_FIELD], public[claims.HISTORY_FIELD] = claims.capability(), []
    if rows > 1:
        query = {'document_id': originals[0]['document_id'], 'record_kind':'fixture',
            'source_match_id':None, 'team_name':'构造甲队', 'team_match':'normalized_exact', 'offset':0, 'limit':20}
        page = project_query(archive, originals, query, 'resolve-identity')
        originals.append(page)
        public['documents'].append(L['_public_page'](page))
    rebind(archive, public, originals)
    public = D['bind'](public, selector)
    assert public['fixture_identity']['resolution']['status'] == 'resolved'
    return selector, archive, public, originals


def rebind(archive, public, originals):
    public['fixture_identity'] = prepare_fixture_identity(archive, public, originals, include_local=True)


def response(public, **updates):
    item = D['response'](public)
    item.update(action='query_acquired', url=None, query=None, engine=None,
        local_query={'document_id':public['documents'][0]['document_id'], 'record_kind':'odds',
            'source_match_id':'1001', 'team_name':None, 'team_match':'exact', 'offset':0, 'limit':20})
    item['local_query'].update(updates)
    return item


def choose(selector, public, *responses):
    batch = D['candidates'](public, *(responses or [response(public)]))
    selection, diag = D['choose'](selector, batch)
    return batch, selection, diag


def test_optional_local_query_can_win_with_zero_http_and_remains_unmeasured(tmp_path):
    policy, _, public, _ = setup(tmp_path)
    public['remaining']['http_calls'] = 0
    before = deepcopy(public)
    batch, choice, diag = choose(policy, public, D['response'](public), response(public))
    assert choice.candidate_id == batch.candidates[1].candidate_id
    assert diag['selection_basis'] == exploration.BASIS
    route = diag['state_bound_voi']['discovered_routes'][choice.candidate_id]
    assert route['source_id'] == 'local:spftll' and route['cost'] == .05
    assert route['origin']['new_source_observation'] is False
    assert route['origin']['observed_at'] == public['documents'][0]['observed_at']
    assert diag['selected_value']['expected_value'] is None
    assert diag['selected_value']['net_value'] is None
    _, _, record = D['accepted'](public, batch, choice)
    assert record['declared_cost'] == .05 and record['numeric_voi_training_eligible'] is False
    assert public == before


def test_cost_changes_choice_without_mandatory_odds_priority(tmp_path):
    policy, _, public, _ = setup(tmp_path, profile={**PROFILE,'query_acquired':.2})
    batch, choice, diag = choose(policy, public, response(public), D['response'](public))
    assert choice.candidate_id == batch.candidates[1].candidate_id
    assert diag['selection_basis'] == exploration.BASIS


@pytest.mark.parametrize('resource,value', [('local_reads',0),('llm_calls',1)])
def test_operational_budget_blocks_local_read(tmp_path, resource, value):
    policy, _, public, _ = setup(tmp_path)
    public['remaining'][resource] = value
    _, choice, _ = choose(policy, public)
    assert choice.candidate_id is None


@pytest.mark.parametrize('field,value', [('seconds_to_deadline',0),('final_decision_only',True)])
def test_deadline_and_terminal_boundary_block_read(tmp_path, field, value):
    policy, _, public, _ = setup(tmp_path)
    public[field] = value
    assert choose(policy, public)[1].candidate_id is None


def test_declared_cost_budget_still_applies_to_local_processing(tmp_path):
    policy, _, public, _ = setup(tmp_path, max_cost=.01)
    _, choice, diag = choose(policy, public)
    assert choice.candidate_id is None
    assert any('declared_cost_allowance_exceeded' in r for r in diag['bounded_exploration']['exclusions'].values())


def test_failed_or_pending_read_cannot_reset_by_page_size_or_question(tmp_path):
    policy, _, public, _ = setup(tmp_path)
    batch, selected, _ = choose(policy, public, response(public, limit=2))
    op, accepted, record = D['accepted'](public, batch, selected)
    op['status'] = 'pending'
    public = D['bind'](public, policy, operations=[op], accepted=[accepted])
    candidate = response(public, limit=20)
    candidate['unknown_to_resolve'] = 'A differently worded question'
    _, choice, diag = choose(policy, public, candidate)
    assert choice.candidate_id is None
    assert 'normalized_action_already_accepted' in next(iter(diag['bounded_exploration']['exclusions'].values()))
    assert discovery.action_key(candidate) == discovery.action_key(accepted)
    assert public[exploration.STATE_FIELD]['reserved_actions'] == 1


def test_paging_uses_actual_next_offset_and_stops_at_verified_end(tmp_path):
    _, archive, public, originals = setup(tmp_path)
    item = response(public, limit=2)
    for offset in (0,2,4):
        item['local_query']['offset'] = offset
        origin = local.bind(public, item)
        assert origin['offset'] == offset
        page = project_query(archive, originals, item['local_query'], 'page-'+str(offset))
        originals.append(page)
        public['documents'].append(L['_public_page'](page))
        rebind(archive, public, originals)
        if offset == 0:
            with pytest.raises(ValueError, match='NEXT_OFFSET'):
                local.bind(public, response(public, offset=20))
    with pytest.raises(ValueError, match='QUERY_COMPLETE'):
        local.bind(public, response(public, offset=6))
    with pytest.raises(ValueError, match='QUERY_COMPLETE'):
        local.bind(public, response(public, limit=1))


@pytest.mark.parametrize('updates', [
    {'source_match_id':'1002'}, {'source_match_id':None}, {'record_kind':'all'},
    {'record_kind':'fixture'}, {'team_name':'构造甲队'}, {'team_match':'contains'},
    {'offset':1}, {'offset':True}, {'limit':True}, {'limit':21},
    {'document_id':'doc-00000000-0000-0000-0000-000000000000'}])
def test_wrong_target_scope_page_and_query_shape_are_excluded(tmp_path, updates):
    policy, _, public, _ = setup(tmp_path)
    batch, choice, diag = choose(policy, public, response(public, **updates))
    assert choice.candidate_id is None
    assert batch.candidates[0].candidate_id in diag['state_bound_voi']['discovery_exclusions']


def test_canonical_input_without_source_identity_does_not_grant_local_match_scope(tmp_path):
    _, archive, public, originals = setup(tmp_path)
    public['match_input'] = 'spftll:9999'
    rebind(archive, public, originals)
    with pytest.raises(ValueError, match='RESOLVED_TARGET'):
        local.bind(public, response(public, source_match_id='9999'))


def test_full_visible_parent_is_not_new_information(tmp_path):
    policy, _, public, _ = setup(tmp_path, rows=1)
    batch, choice, diag = choose(policy, public)
    assert choice.candidate_id is None
    assert diag['state_bound_voi']['discovery_exclusions'][batch.candidates[0].candidate_id] == 'LOCAL_ACQUISITION_PARENT_ALREADY_VISIBLE'


def test_post_kickoff_snapshot_is_not_prematch_evidence(tmp_path):
    _, archive, public, originals = setup(tmp_path)
    public['snapshot_as_of'] = '2027-10-01T12:00:00Z'
    rebind(archive, public, originals)
    with pytest.raises(ValueError, match='PREMATCH_OBSERVATION'):
        local.bind(public, response(public))


@pytest.mark.parametrize('where', ['parent', 'page'])
def test_forged_public_projection_requires_independent_bridge_binding(tmp_path, where):
    _, _, public, _ = setup(tmp_path)
    doc = public['documents'][0 if where=='parent' else -1]
    doc['observed_at'] = '2020-01-01T00:00:00Z'
    with pytest.raises(ValueError, match='FIXTURE_IDENTITY_BRIDGE'):
        local.bind(public, response(public))


@pytest.mark.parametrize('cost', [-1, True, float('nan'), float('inf'), 10**400, '.1'])
def test_local_cost_validation(cost):
    with pytest.raises(ValueError, match='DISCOVERY_FINITE'):
        discovery.configuration({**PROFILE,'query_acquired':cost}, 'loss')


def test_opt_in_profile_identity_and_old_configuration_are_preserved(tmp_path):
    policy, _, public, _ = setup(tmp_path)
    new = discovery.descriptor(policy_identity(policy))
    assert new['version'] == discovery.LOCAL_VERSION
    assert new['local_binding_version'] == local.VERSION
    old = D['policy']()
    assert discovery.descriptor(policy_identity(old))['version'] == discovery.VERSION
    assert 'query_acquired' not in old.configuration['discovered_routes']
    old_public = D['bind'](public, old)
    assert choose(old, old_public)[1].candidate_id is None
    disabled = D['policy'](profile={**PROFILE,'query_acquired':None})
    _, choice, diag = choose(disabled, D['bind'](public, disabled))
    assert choice.candidate_id is None
    assert next(iter(diag['state_bound_voi']['discovery_exclusions'].values())) == 'DISCOVERY_LOCAL_COST_UNDECLARED'


def test_local_only_profile_can_be_zero_cost_but_not_positive_value(tmp_path):
    policy, _, public, _ = setup(tmp_path, profile={'search_web':{},'read_url':None,'query_acquired':0})
    _, choice, diag = choose(policy, public)
    assert choice.candidate_id is not None and diag['selected_value']['cost'] == 0
    assert diag['selected_value']['expected_value'] is None


def test_real_proposer_protocol_exposes_optional_local_action_in_one_call(tmp_path, monkeypatch):
    from jsonschema import validate
    from football_agent.acquisition_boundary import SHARED_FIELDS, OPTIONAL_SHARED_FIELDS
    from football_agent.experience import validate_decision
    from football_agent.match_proposals import MatchCandidateProvider, MatchLocalProposalProtocol
    policy, _, public, _ = setup(tmp_path)
    item = response(public)
    fields = SHARED_FIELDS + tuple(k for k in OPTIONAL_SHARED_FIELDS if k in item)
    proposal = {k:item[k] for k in fields}
    proposal.update(candidates=[{k:v for k,v in item.items() if k not in fields and k!='alternatives'}],
                    recommended_index=0)
    seen = []
    class Provider:
        def __init__(self, folder, *, protocol, **kwargs):
            self.protocol = protocol
        def decide(self, snapshot):
            validate(proposal, self.protocol.schema)
            seen.append((snapshot, self.protocol))
            return {'response':proposal, 'test_double':True}
    monkeypatch.setattr('football_agent.match_proposals.ExperienceProvider', Provider)
    generated = MatchCandidateProvider(tmp_path).propose(public)
    assert len(seen) == 1 and generated['candidates'][0]['local_query'] == item['local_query']
    assert seen[0][1].version.endswith('-resolved-local-v1')
    validate_decision(generated['candidates'][0], public)
    assert 'OPTIONAL RESOLVED-MATCH' not in MatchLocalProposalProtocol.prompt
    assert choose(policy, public, *generated['candidates'])[1].candidate_id is not None
