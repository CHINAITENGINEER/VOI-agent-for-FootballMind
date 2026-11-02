"""Fixed crossed acquisition mechanism, not real football performance.

48 independent constructed B fixtures, 3 dates, 2 visible contexts and 2 fixed
same-host/same-tool/same-cost candidates; twelve observations per cell. Both
candidate titles are visible in both states. Only the selected response is
observed. Every signed target is recomputed by the same actual frozen f.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from football_agent.learned_voi_web import _web_action
from football_research.acquisition.context_gain import _signed_records, _query_input, _digest, VERSION
from football_research.acquisition.context_predictor import ContextPredictor
from football_research.acquisition.context_value import ContextValueModel, PARAMETERS, _public
from football_research.acquisition.visible_state import encode_visible_state
from test_context_gain import transition, save
from test_context_predictor import records, AS_OF, WORDS
from test_context_value import ConstructedFitClock, G_AS_OF


STATES = ['已知状态：主力前锋出场情况未明，球队组织进攻需要确认。',
          '已知状态：场地天气情况未明，球队比赛条件需要确认。']
LABELS = ['官方阵容消息与球员出场说明', '官方天气消息与球场情况说明']
URLS = ['https://www.uefa.com/news/lineup-context/', 'https://www.uefa.com/news/weather-context/']


def article(ident, text, stamp, *, available=None):
    return {'document_id': ident, 'raw_sha256': 'a'*64, 'url': 'https://www.uefa.com/news/'+ident+'/',
        'status': 'OK', 'kind': 'article', 'text': text, 'links': [], 'source_published_at': None,
        'observed_at': stamp, 'available_at': available or stamp,
        'structured_news': {'version': 'structured-news-discovery-v3', 'role': 'article',
            'segments': [{'text': text, 'start': 0, 'end': len(text),
                'content_source': 'json_ld_article_body', 'json_pointer': 'script[0].articleBody'}]}}


def visible_pair(physical, state_index, candidate_index, *, month=1, day=20):
    at = datetime(2027, month, day, 10, tzinfo=timezone.utc)
    actions = [_web_action('read_url', url, cost={'money': .01, 'requests': 1.}) for url in URLS]
    parent = article('already-read-state', STATES[state_index], at.isoformat())
    parent['links'] = [{'url': url, 'title': label,
        'observed_anchors': [{'anchor_index': i, 'text': label, 'text_truncated': False}],
        'observed_anchors_truncated': False} for i, (url, label) in enumerate(zip(URLS, LABELS))]
    before = {'state': {'match_id': 'provider:'+physical,
        'prediction_cutoff': (at+timedelta(hours=8)).isoformat(),
        'base': {'kickoff': (at+timedelta(hours=9)).isoformat()}, 'action_history': []},
        'decision_at': (at+timedelta(seconds=1)).isoformat(), 'documents': [parent], 'attempts': [],
        'prediction': {'probabilities': [.45, .3, .25], 'probability_basis': 'observed_market_snapshot'}}
    action = actions[candidate_index]
    after = deepcopy(before)
    after['decision_at'] = (at+timedelta(seconds=4)).isoformat()
    after['state']['action_history'] = [{'action_id': action['action_id']}]
    after['attempts'] = [{'action_id': action['action_id'], 'status': 'OK', 'information_key': action['information_key']}]
    after['documents'].append(article('actual-response', WORDS[0 if state_index == candidate_index else 2],
        (at+timedelta(seconds=2)).isoformat(), available=(at+timedelta(seconds=3)).isoformat()))
    return encode_visible_state(before, actions), encode_visible_state(after), action


def mechanism(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    f = ContextPredictor().fit(records(), as_of=AS_OF, dataset_version='constructed-crossed-f',
                               evidence_origin='constructed_mechanism')
    fpath = root/'f.json'
    f.save(fpath)
    frozen = f, fpath, sha256(fpath.read_bytes()).hexdigest()
    prefixes, pairs, plans = [], [], {}
    for s in range(2):
        for q in range(2):
            for replica in range(12):
                physical = f'crossed-B-{s}-{q}-{replica}'
                day = 20+replica % 3
                block, pair, plan = transition(frozen, physical=physical, outcome=0)
                block, pair, plan = json.loads(json.dumps([block, pair, plan]).replace('2027-01-20', f'2027-01-{day:02}'))
                before, after, action = visible_pair(physical, s, q, day=day)
                block[0]['visible_state'], block[1]['visible_state'] = before, after
                pair[0]['action'] = action
                prefixes.extend(block); pairs.extend(pair); plans.update(plan)
    features, targets, excluded = _signed_records(prefixes, pairs, plans, model=f,
        model_sha=frozen[2], as_of=G_AS_OF, origin='constructed_mechanism')
    assert not excluded
    folder = root/'gain'
    folder.mkdir()
    source = {'schema': 'constructed-fixed-crossed-response-mechanism-v1', 'input_bindings': {},
              'prefixes': prefixes, 'transitions': pairs, 'design': {'states': STATES, 'labels': LABELS,
                  'cell_replications': 12, 'kickoff_dates': 3, 'fixed_response_rule': 'home_body_when_state_equals_query_else_away_body'}}
    for name, value in [('features', features), ('targets', targets), ('source-inputs', source),
                        ('calibration-registrations', list(plans.values()))]:
        save(folder/(name+'.json'), value)
    save(folder/'report.json', {'schema': VERSION, 'as_of': G_AS_OF, 'evidence_origin': 'constructed_mechanism',
        'predictor_sha256': frozen[2], 'predictor_model_version': f.model_version,
        'features_sha256': _digest(features), 'targets_sha256': _digest(targets),
        'source_inputs_file_sha256': sha256((folder/'source-inputs.json').read_bytes()).hexdigest(),
        'registrations_file_sha256': sha256((folder/'calibration-registrations.json').read_bytes()).hexdigest()})
    with patch('football_research.acquisition.context_value.datetime', ConstructedFitClock):
        g = ContextValueModel().fit(folder, predictor_path=fpath, as_of=G_AS_OF, dataset_version='constructed-fixed-crossed-B')
    g.save(root/'g.json')
    values, additive, priors, support = [], [], [], []
    for s in range(2):
        row, old, p, supported = [], [], [], []
        for q in range(2):
            before, _, action = visible_pair('heldout-'+str(s), s, q, month=2, day=10)
            predicted = f.predict(before, physical_match_id='heldout-'+str(s))['probabilities']
            inputs = _query_input(before, action, predicted)
            result = g.predict(inputs, physical_match_id='heldout-'+str(s), predictor_sha256=frozen[2])
            vector = g._vector(_public(inputs))[0]
            count = PARAMETERS['interaction_size']
            old.append(float(vector[:-count]@np.array(g.coefficients[:-count])))
            row.append(result['expected_signed_gain']); supported.append(result['support_status']); p.append(predicted)
        values.append(row); additive.append(old); priors.append(p); support.append(supported)
    return {'evidence_origin': 'constructed_mechanism', 'physical_B_matches': 48,
        'states': STATES, 'candidate_labels': LABELS, 'candidate_urls': URLS,
        'fixed_gain_generation': 'same_f_actual_before_after_log_loss_difference_never_handwritten_gain',
        'forecast_before': priors, 'estimated_signed_gain': values, 'support': support,
        'additive_terms_only': additive, 'additive_comparison_scope': 'same_fit_terms_structural_diagnostic_not_retrained_baseline',
        'preferred_candidates': [int(np.argmax(v)) for v in values],
        'positive_target_count': sum(t['signed_gain'] > 0 for t in targets),
        'negative_target_count': sum(t['signed_gain'] < 0 for t in targets),
        'real_world_benefit_proven': False, 'network_requests': 0}


def test_same_candidates_reverse_ranking_when_only_known_context_changes(tmp_path):
    result = mechanism(tmp_path)
    assert result['support'] == [['supported', 'supported'], ['supported', 'supported']]
    assert result['preferred_candidates'] == [0, 1]
    assert all(p == [.45, .3, .25] for row in result['forecast_before'] for p in row)
    assert result['positive_target_count'] == result['negative_target_count'] == 24
    old = result['additive_terms_only']
    assert old[0][0]-old[0][1] == pytest.approx(old[1][0]-old[1][1], abs=1e-12)
