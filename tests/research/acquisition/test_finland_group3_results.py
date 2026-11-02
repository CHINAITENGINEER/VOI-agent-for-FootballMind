"""Constructed lower-stage result fixtures; never real samples or HTTP."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
import socket

import pytest

from football_research.data_adapters import finland_results as results
from football_research.data_adapters.finland_result_capture import (
    capture_finland_result, detail_url, verify_capture_receipts,
)
from football_research.data_adapters.official_fixture import prepare_official_fixture
from football_research.data_adapters.official_stage_registry import GROUP3_TEAMS
from football_research.data_adapters.verified_fixtures import verify_independent_settlement
from football_spider.storage.raw_archive import RawArchive
from test_official_fixture_results import BEFORE, Source

AS_OF = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail('Group 3 result tests must use constructed transport only')
    monkeypatch.setattr(socket, 'getaddrinfo', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    monkeypatch.setattr(socket.socket, 'connect', deny)


class Group3Source(Source):
    def __init__(self):
        super().__init__()
        self.home, self.away = 'FC Lahti', 'TPS'
        self.schedule_mutation = self.terminal_mutation = None

    def schedule(self, score='-'):
        body = super().schedule(score).decode()
        body = body.replace('value="2" selected>Mestaruussarja', 'value="3" selected>Karsintasarja')
        teams = ''.join(f'<option value="{team["native_id"]}">{name}</option>'
                        for name, team in GROUP3_TEAMS.items())
        body = re.sub(r'<select name="team">.*?</select>', '<select name="team">'+teams+'</select>', body)
        return self.schedule_mutation(body.encode()) if self.schedule_mutation else body.encode()

    def terminal(self):
        body = super().terminal()
        return self.terminal_mutation(body) if self.terminal_mutation else body


def prepared(root):
    source = Group3Source()
    plan = prepare_official_fixture('veikkausliiga:2026:match:'+source.native,
        output=root/'identity', clock=lambda:BEFORE, transport=source, schedule_group='3')
    assert plan['status'] == 'ready', plan
    binding = json.loads((root/'identity/identity.json').read_text(encoding='utf-8'))
    assert binding['schedule_group'] == '3'
    assert binding['canonical']['stage'] == 'Karsintasarja'
    assert binding['canonical']['home_id'] == 'veikkausliiga:team:60969'
    assert binding['canonical']['away_id'] == 'veikkausliiga:team:7'
    return source, binding, root/'identity/identity-archive'


def capture(root, source, binding, identity_archive, *, at=AS_OF):
    return capture_finland_result(binding, identity_archive_root=identity_archive,
        output=root/'result', clock=lambda:at, transport=source)


def test_group3_finished_two_halves_settles_with_distinct_rule_and_two_actual_result_requests(tmp_path):
    source, binding, identity_archive = prepared(tmp_path)
    check = capture(tmp_path, source, binding, identity_archive)
    assert check['status'] == 'VERIFIED'
    assert check['schema'] == results.VERSION
    assert check['regulation_rule']['version'] == results.GROUP3_RULE_VERSION
    assert check['regulation_rule']['scope'] == '2026_senior_men_veikkausliiga_karsintasarja_observed_two_halves'
    row, = check['settlement_rows']
    assert (row['home_goals'], row['away_goals']) == (2, 0)
    assert row['regulation_rule_version'] == results.GROUP3_RULE_VERSION
    assert row['score_period'] == 'regulation' and row['available_at'] == AS_OF.isoformat()
    assert row['match_id'] == binding['canonical']['source_event_id']
    assert [call['stage'] for call in source.calls] == ['official_schedule', 'schedule', 'detail']
    assert source.calls[1]['params'] == {'group':'3'}
    assert source.calls[2]['url'] == detail_url(binding) and source.calls[2]['params'] == {}
    receipts = verify_capture_receipts(binding, tmp_path/'result', as_of=AS_OF)
    assert receipts['http_requests_actual'] == 2 and len(receipts['cached_rule_observation_ids']) == 3
    archive = RawArchive(tmp_path/'result')
    assert verify_independent_settlement(binding, identity_archive_root=identity_archive,
        result_archive_root=archive.root, observation_ids=[o.id for o in archive.observations()], as_of=AS_OF) == check
    assert 'p0' not in json.dumps(check) and 'initialization_manifest' not in json.dumps(check)


def test_group3_reprojects_original_rule_bytes_without_changing_manifest_or_receipt_clocks(tmp_path):
    original = RawArchive(results.RULE_PACKAGE_ROOT)
    before = {str(p.relative_to(original.root)):sha256(p.read_bytes()).hexdigest()
              for p in original.root.rglob('*') if p.is_file()}
    old = results.verify_rule_package(as_of=AS_OF)
    assert old == results.verify_rule_package(as_of=AS_OF, schedule_group='2')
    new = results.copy_rule_observations(tmp_path/'copy', as_of=AS_OF, schedule_group='3')
    assert new['status'] == 'VERIFIED' and new['rule_version'] == results.GROUP3_RULE_VERSION
    assert new['observation_ids'] == old['observation_ids'] and new['http_requests'] == 0
    assert new['manifest_sha256'] == old['manifest_sha256']
    assert [o.model_dump() for o in RawArchive(tmp_path/'copy').observations()] == [o.model_dump() for o in original.observations()]
    assert before == {str(p.relative_to(original.root)):sha256(p.read_bytes()).hexdigest()
                      for p in original.root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('phrase', [
    'Kuuden joukkueen yksinkertainen Karsintasarja (15 ottelua)',
    'Yhteensä Veikkausliigassa pelataan 177 ottelua',
    'Lisäksi pelataan kaksi ottelua Veikkausliigan karsintoja',
    'Karsinnat eivät ole osa Veikkausliigaa',
])
def test_group3_rule_requires_the_league_membership_and_separate_playoff_semantics(tmp_path, phrase):
    source = RawArchive(results.RULE_PACKAGE_ROOT)
    original = next(o for o in source.observations() if o.endpoint == results.FORMAT_URL)
    body = source.read_body(original)
    # Construct a response from the readable text, then remove one required
    # assertion. This is projection rejection, not an accepted rule snapshot.
    from bs4 import BeautifulSoup
    text = ' '.join(BeautifulSoup(body, 'html.parser').find('main').get_text(' ', strip=True).split())
    assert phrase in text
    body = ('<html><main>'+text.replace(phrase, '')+'</main></html>').encode()
    archive = RawArchive(tmp_path/'raw')
    observation = archive.save('open_page', results.FORMAT_URL, {}, body, status=200, observed_at=AS_OF,
        context={'requested_at':AS_OF.isoformat()})
    projection = results._rule_projection(observation, body, as_of=AS_OF, schedule_group='3')
    assert projection['status'] == 'UNCONFIRMED'


def test_group3_not_due_and_live_results_remain_pending(tmp_path):
    source, binding, identity_archive = prepared(tmp_path/'not-due')
    check = capture(tmp_path/'not-due', source, binding, identity_archive, at=BEFORE)
    assert check['status'] == 'PENDING' and check['settlement_rows'] == []
    assert len(source.calls) == 1
    assert verify_capture_receipts(binding, tmp_path/'not-due/result', as_of=BEFORE)['http_requests_actual'] == 0
    source, binding, identity_archive = prepared(tmp_path/'live')
    source.finished = False
    source.score = '-'
    check = capture(tmp_path/'live', source, binding, identity_archive)
    assert check['status'] == 'PENDING' and check['settlement_rows'] == [] and len(source.calls) == 3
    assert any(p['reason'] == 'official_karsintasarja_score_not_available' for p in check['projections'])


@pytest.mark.parametrize('old,new', [
    ('head finished', 'head live'),
    ("90'", "105'"),
    ('Puoliaika alkoi', 'Jatkoaika alkoi'),
    ('Puoliaika päättyi', 'Rangaistuspotkukilpailu päättyi'),
    ('<p class="info center">Ottelu päättyi</p>', ''),
    ('<span id="home-goals">2</span>', '<span id="home-goals">2.0</span>'),
    ('<span id="home-goals">2</span>', '<span id="home-goals">3</span>'),
    ('FC Lahti</a>', 'FF Jaro</a>'),
])
def test_group3_terminal_state_period_score_and_order_are_required(tmp_path, old, new):
    source, binding, identity_archive = prepared(tmp_path)
    assert old.encode() in source.terminal()
    source.terminal_mutation = lambda body:body.replace(old.encode(), new.encode())
    check = capture(tmp_path, source, binding, identity_archive)
    assert check['status'] != 'VERIFIED' and check['settlement_rows'] == []


@pytest.mark.parametrize('old,new', [
    ('value="3" selected>Karsintasarja', 'value="2" selected>Mestaruussarja'),
    ('value="3" selected>Karsintasarja', 'value="3" selected>Karsinnat'),
    ('9876543/seuranta/', '1234567/seuranta/'),
    ('2 — 0', '2 — 0 (penalties 5-4)'),
    ('FC Lahti - TPS', 'TPS - FC Lahti'),
])
def test_wrong_stage_or_unobserved_detail_link_never_causes_a_second_request(tmp_path, old, new):
    source, binding, identity_archive = prepared(tmp_path)
    source.schedule_mutation = lambda body:body.replace(old.encode(), new.encode())
    check = capture(tmp_path, source, binding, identity_archive)
    assert check['status'] != 'VERIFIED' and check['settlement_rows'] == []
    assert [call['stage'] for call in source.calls] == ['official_schedule', 'schedule']
    assert not (tmp_path/'result/requests/detail.intent.json').exists()
    assert verify_capture_receipts(binding, tmp_path/'result', as_of=AS_OF)['http_requests_actual'] == 1


def test_group3_schedule_requires_explicit_opt_in_and_matching_parameters(tmp_path):
    source, binding, identity_archive = prepared(tmp_path)
    archive = RawArchive(tmp_path/'raw')
    body = source.schedule('2 — 0')
    observation = archive.save('open_page', results.RESULTS_URL, {'group':'3'}, body, status=200,
        observed_at=AS_OF, context={'requested_at':AS_OF.isoformat()})
    assert results.project_championship_schedule(observation, body, binding['canonical'], as_of=AS_OF)['status'] == 'IGNORED'
    assert results.project_championship_schedule(observation, body, binding['canonical'], as_of=AS_OF,
        schedule_group='3')['status'] == 'SCORE'
    changed = observation.model_copy(update={'params':{'group':'2'}})
    assert results.project_championship_schedule(changed, body, binding['canonical'], as_of=AS_OF,
        schedule_group='3')['status'] == 'IGNORED'
    canonical = {**binding['canonical'], 'stage':'Mestaruussarja'}
    assert results.project_championship_schedule(observation, body, canonical, as_of=AS_OF,
        schedule_group='3')['status'] == 'UNCONFIRMED'


@pytest.mark.parametrize('change', ['missing_group', 'group2', 'unsupported_group', 'wrong_stage', 'legacy_schema'])
def test_group3_followup_rejects_missing_or_incompatible_scope(tmp_path, change):
    _, binding, _ = prepared(tmp_path)
    binding = deepcopy(binding)
    if change == 'missing_group':
        binding.pop('schedule_group')
    elif change == 'group2':
        binding['schedule_group'] = '2'
    elif change == 'unsupported_group':
        binding['schedule_group'] = '4'
    elif change == 'wrong_stage':
        binding['canonical']['stage'] = 'Karsinnat'
    else:
        binding['schema'] = 'finland-stage-fixture-identity-v1'
    with pytest.raises(ValueError, match='scope|group'):
        results.result_schedule_group(binding)
    with pytest.raises(ValueError, match='scope|group'):
        detail_url(binding)


def test_group3_receipts_reject_changed_group_after_successful_capture(tmp_path):
    source, binding, identity_archive = prepared(tmp_path)
    assert capture(tmp_path, source, binding, identity_archive)['status'] == 'VERIFIED'
    path = tmp_path/'result/capture-plan.json'
    plan = json.loads(path.read_text(encoding='utf-8'))
    plan['schedule_params'] = {'group':'2'}
    path.write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest'):
        verify_capture_receipts(binding, tmp_path/'result', as_of=AS_OF)


@pytest.mark.parametrize('group', [3, True, None, '1', '03'])
def test_rule_group_requires_registered_explicit_string(group):
    with pytest.raises(ValueError, match='group'):
        results.verify_rule_package(as_of=AS_OF, schedule_group=group)
