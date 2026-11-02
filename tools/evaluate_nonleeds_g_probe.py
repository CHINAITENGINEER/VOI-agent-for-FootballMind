"""New official-body historical probe of frozen f/g, without fitting on its label.

The small source review below is authored by the current Codex session. It is
not a general extractor, an API response, or an independently verified truth set.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import socket
import sys
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_reviewed_rich_f import read, write, history, row_for
from tools.train_state_candidate_g import before_snapshot
from tools.reconstruct_panel_priors import TARGETS, no_network
from football_research.acquisition.match_conditioned_f import ReconstructedMatchPrior, bind_prior
from football_research.acquisition.historical_context_training import _discovery_panel, _project
from football_research.acquisition.news_link_projection import VERSION as PROJECTION
from football_research.acquisition.entity_semantic_f import EntityEvidenceBank
from football_research.acquisition.semantic_f import EXTRACTION_VERSION
from football_research.acquisition.context_predictor import _digest
from football_research.acquisition import entity_evidence_updates as f
from football_research.acquisition import scoped_reviewed_f as scoped
from football_research.acquisition import reviewed_rich_f as rich
from football_research.acquisition import historical_gain_head as gain
from football_research.acquisition import structured_state_gain as g
from football_research.acquisition.historical_policy_comparison import rollout

MATCH = 'understat:28900'
F_PATH = ROOT / 'var/reviews/fm-historical-transition-g-training-20261014-v2/f-labeler.json'
G_DIR = ROOT / 'var/reviews/fm-g-structured-state-20261014-v1'


def reviewed_claim(text, entity, team, concept, quote, statement, *, value=None,
                   scope='this_match', context=None):
    """Require verbatim, uniquely located support; assign no effect or gain."""
    if text.count(quote) != 1:
        raise ValueError('Review quote must have one exact location')
    quotes = [quote]
    if context is not None and context != quote:
        if text.count(context) != 1:
            raise ValueError('Review context must have one exact location')
        quotes.append(context)
    claim = {'team': team, 'entity': entity, 'kind': 'availability' if value else 'other',
             'value': value or 'unmodeled', 'scope': scope, 'statement': statement, 'quotes': quotes}
    return {'claim': claim, 'concept': concept, 'spans': [
        {'start': text.index(q), 'end': text.index(q)+len(q), 'quote': q} for q in quotes],
        'information_value': None, 'automatic_truth_label': False}


def review_documents(ledger, source):
    reviews = {}
    for doc in ledger:
        body_bytes = (source/doc['body_path']).read_bytes()
        assert sha256(body_bytes).hexdigest() == doc['body_sha256']
        text = body_bytes.decode('utf-8')
        assert text == '\n'.join(doc['paragraphs'])
        p = doc['paragraphs']
        claim = lambda *args, **kwargs: reviewed_claim(text, *args, **kwargs)
        if doc['id'] == 'C01':
            items = [
                claim('Cole Palmer', 'home', 'confirmed_absence', p[3],
                      'Maresca says Cole will not feature against Barcelona or Arsenal because of a fractured toe.',
                      value='unavailable', context=p[1]),
                claim('Benoit Badiashile', 'home', 'other_fixture_availability', p[9],
                      'Badiashile is available for the next-day Burnley match, not a statement about Arsenal.',
                      scope='historical', context=p[6]),
                claim('Enzo Fernandez', 'home', 'other_fixture_availability', p[10],
                      'Fernandez is available for tomorrow’s Burnley match; no Arsenal availability inferred.',
                      scope='historical', context=p[7]),
                claim('Pedro Neto', 'home', 'other_fixture_availability', p[10],
                      'Neto is available for tomorrow’s Burnley match; no Arsenal availability inferred.',
                      scope='historical', context=p[7]),
            ]
            title_items = []
        elif doc['id'] == 'C02':
            items = [
                claim('Cole Palmer', 'home', 'confirmed_available', p[2],
                      'Palmer is available for the Arsenal game; a start is possible, not confirmed.',
                      value='available'),
                claim('Dario Essugo', 'home', 'partial_training', p[4],
                      'Essugo took part in the morning session; match availability is not established.',
                      scope='team_context'),
                claim('Cole Palmer', 'home', 'fitness_not_full', p[9],
                      'Maresca says Palmer needs time to be 100 per cent fit, despite being available.',
                      scope='team_context'),
                claim('Chelsea', 'home', 'conditional_pairing', p[10],
                      'Estevao and Palmer can play together depending on opponent and game plan; no lineup commitment.',
                      scope='team_context'),
                claim('Chelsea', 'home', 'match_ambition', p[14],
                      'Chelsea will try to beat Arsenal; this is ambition, not a tactical attacking plan.',
                      scope='team_context'),
            ]
            title_items = [reviewed_claim(doc['title'], 'Cole Palmer', 'home', 'confirmed_available',
                doc['title'], 'The headline explicitly says Cole Palmer is fit for Arsenal.', value='available')]
        elif doc['id'] == 'C03':
            items = [
                claim('Cole Palmer', 'home', 'confirmed_bench', p[3],
                      'Cole Palmer is named on the bench against Arsenal; selection is not full medical fitness.'),
                claim('Chelsea starting XI', 'home', 'confirmed_starter', p[4],
                      'The source supplies Chelsea’s starting XI as one collective selection assertion.'),
                claim('Arsenal starting XI', 'away', 'confirmed_starter', p[6],
                      'The source supplies Arsenal’s starting XI as one collective selection assertion.'),
                claim('Chelsea substitutes', 'home', 'confirmed_bench', p[5],
                      'The source supplies Chelsea’s substitutes as one collective selection assertion.', context=p[4]),
                claim('Arsenal substitutes', 'away', 'confirmed_bench', p[7],
                      'The source supplies Arsenal’s substitutes as one collective selection assertion.', context=p[6]),
            ]
            title_items = []
        else:
            raise ValueError('Unreviewed document')
        reviews[doc['id']] = {'body': items, 'title': title_items,
            'reviewer': 'current Codex session; exact serving model ID not exposed',
            'body_read_in_full': True, 'review_scope': 'salient football assertions, not exhaustive sentence annotation',
            'excluded_content': ['prior Barcelona/Burnley results are not target results',
                                 'charity appeals and app promotion retained in body, not match evidence'],
            'selection_lists': 'collective assertions plus explicitly narrated Palmer; no invented per-player medical clearance',
            'historical_version_unknown': True}
    return reviews


def enrich_row(row, bank, fixture, clock_by_url, concepts):
    events = {}
    for input_id in row['used_input_ids']:
        rec = bank.records[input_id]
        for claim in rec['response']['claims']:
            event = {'physical_match_id': row['physical_match_id'], 'input_id': input_id,
                'entity': claim['entity'], 'document_url': rec['input']['url'],
                'source_host': urlsplit(rec['input']['url']).hostname,
                'source_published_at': clock_by_url[rec['input']['url']], 'historical_available_at': None,
                'claim': claim, 'evidence': rich.evidence_tokens(claim, rec['input']['channel'], fixture,
                                                               concepts[input_id, claim['statement']])}
            eid = _digest(event)
            events[eid] = {**event, 'event_id': eid}
    assert rich.unique_evidence([x['evidence'] for x in events.values()]) == row['evidence']
    return {**row, 'evidence_events': list(events.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    socket.socket.connect = no_network
    socket.getaddrinfo = no_network
    ledger = read(args.source/'ledger.json')
    assert len(ledger) == 3 and all(x['status'] == 'BODY_RECOVERED' for x in ledger)
    reviews = review_documents(ledger, args.source)
    write(out/'semantic-review.json', reviews)
    labeler = f.load(F_PATH)
    models = {name: g.load(G_DIR/f'g-{name}.json') for name in ('baseline', 'combined')}
    bindings = {str(p): sha256(p.read_bytes()).hexdigest()
                for p in (F_PATH, G_DIR/'g-baseline.json', G_DIR/'g-combined.json')}
    assert MATCH not in labeler['head']['head']['encoder']['fit_match_ids']
    assert all(MATCH not in m['fit_match_ids'] for m in models.values())
    protocol = read(ROOT/'var/f-entity-semantic-20261011-v2/protocol.json')
    excluded = set(sum(protocol['splits'].values(), [])+TARGETS+[MATCH])
    write(out/'plan.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'fixture_id': MATCH, 'models': bindings, 'base_fit_exclusions': sorted(excluded),
        'f_or_g_refit': False, 'g_training_new_rows': False, 'online_selection_allowed': False,
        'budgets': [1, 2, 3], 'cost_loss_units': [0., .001, .01],
        'menu': 'enumerated historical title/body subsets; not actual historical tool trace',
        'title_information': 'explicit Palmer availability in headline enters f before reading body',
        'selection_rule': 'three source URLs frozen before predictions; retain positive/negative/zero transitions',
        'disclosure': 'Search exposed result snippet; analyst is not strictly outcome-blind.',
        'historical_version_unknown': True, 'prospective_training_eligible': False})
    archive, sources = history()
    by_id = {f"{r['source']}:{r['source_id']}": r for r in archive}
    target = by_id[MATCH]
    fixture = {'physical_match_id': MATCH, 'match_id': MATCH,
        'home_team_id': target['home_team_id'], 'away_team_id': target['away_team_id'],
        'home_name': target['home_team_name'], 'away_name': target['away_team_name'],
        'competition_id': target['competition_id'], 'competition': target['competition_name'],
        'kickoff': target['kickoff']}
    assert fixture['home_name'] == 'Chelsea' and fixture['away_name'] == 'Arsenal'
    docs = [{'url': x['url'], 'title': x['title'], 'text': (args.source/x['body_path']).read_text('utf-8'),
             'captured_at': x['save_returned_at'], 'source_published_at': x['source_published_at']}
            for x in ledger]
    case = {'fixture': fixture, 'documents': docs}
    panel = list(_discovery_panel(case))
    generator = ReconstructedMatchPrior([r for k,r in by_id.items() if k not in excluded or k == MATCH],
                                        source_binding={'archives': sources, 'excluded_targets': sorted(excluded)})
    prior = generator.snapshot(fixture, panel[0][-1]['decision_at'])
    assert not set(prior['training_match_ids']) & excluded
    write(out/'prior-snapshot.json', prior)
    full = bind_prior(panel[-1][-1], prior)
    items = EntityEvidenceBank._items(_project(full, text_projection=PROJECTION))
    review_by_url = {x['url']: reviews[x['id']] for x in ledger}
    records, concepts = [], {}
    for channel, inputs in items.items():
        for key, public in inputs.items():
            annotated = review_by_url[public['url']][channel]
            records.append({'input': public, 'response': {'schema': EXTRACTION_VERSION,
                'input_id': key, 'claims': [x['claim'] for x in annotated]},
                'provenance': {'kind': 'explicit_Codex_full_body_review', 'provider_execution': False}})
            for a in annotated:
                concepts[key, a['claim']['statement']] = a['concept']
    bank = EntityEvidenceBank(records)
    write(out/'semantic-records.json', records)
    clocks = {x['url']: x['source_published_at'] for x in ledger}
    states, rows, predictions, feature_audit = {}, {}, {}, {}
    for discovered_mask, acquired_mask, discovered, acquired, state in panel:
        state = bind_prior(state, prior)
        row = row_for(state, bank, concepts, None, None)
        row = enrich_row(row, bank, fixture, clocks, concepts)
        used = sorted(row['used_input_ids'])
        sid = _digest({'match': MATCH, 'inputs': used})
        assert sid not in states
        states[sid] = {'physical_match_id': MATCH, 'used_input_ids': used,
                       'read_count': len(acquired), 'title_count': len(discovered),
                       'discovered_mask': discovered_mask, 'acquired_mask': acquired_mask}
        rows[sid] = row
        predictions[sid] = f.predict(labeler, row).tolist()
        resolved, audit = f.resolve(row)
        routed, routing = scoped.route(resolved, labeler['head']['mode'])
        vector = rich.features(labeler['head']['head']['encoder'], routed)
        feature_audit[sid] = {'resolution': audit, 'routing': routing,
            'nonzero_features': int(np.count_nonzero(vector)), 'features': vector.tolist()}
    write(out/'prediction-input-rows.json', rows)
    write(out/'states.json', states)
    write(out/'fixed-f-predictions.json', predictions)
    write(out/'feature-audit.json', feature_audit)
    # Labels are serialized and joined only after all fixed-model predictions.
    outcome = 0 if target['home_goals'] > target['away_goals'] else 1 if target['home_goals'] == target['away_goals'] else 2
    write(out/'independent-outcome.json', {'physical_match_id': MATCH,
        'home_goals': target['home_goals'], 'away_goals': target['away_goals'], 'outcome': outcome,
        'original_observed_at': target['observed_at'], 'observation_id': target['observation_id'],
        'source_data_version': target['data_version'], 'archives': sources,
        'prediction_rows_receive_outcome': False, 'analyst_blind': False})
    transitions = []
    for sid, s in states.items():
        keys = set(s['used_input_ids'])
        for tid, after in states.items():
            added = set(after['used_input_ids'])-keys
            if not keys <= set(after['used_input_ids']) or len(added) != 1:
                continue
            body = next(iter(added))
            public = bank.records[body]['input']
            if public['channel'] != 'body':
                continue
            titles = [bank.records[k]['input'] for k in keys
                      if bank.records[k]['input']['channel'] == 'title'
                      and bank.records[k]['input']['url'] == public['url']]
            assert len(titles) == 1
            p, q = predictions[sid], predictions[tid]
            query = {'schema': gain.QUERY_SCHEMA, 'probabilities': p,
                     'acquired_documents': s['read_count'], 'visible_candidates': 0,
                     'article_age_hours': (f.timestamp(rows[sid]['decision_at'])-f.timestamp(clocks[public['url']])).total_seconds()/3600,
                     'title': titles[0]['text']}
            transitions.append({'physical_match_id': MATCH, 'state_id': sid, 'next_state_id': tid,
                'action_id': 'read-'+_digest({'url': public['url']})[:20],
                'body_input_id': body, 'body_url': public['url'], 'query': query,
                'prediction_before': p, 'prediction_after': q, 'outcome': outcome,
                'signed_gain': gain.signed_gain(p, q, outcome), 'f_labeler_sha256': bindings[str(F_PATH)],
                'provenance': 'enumerated_historical_reconstruction', 'propensity': None,
                'prospective_training_eligible': False, 'g_split': 'new_nonleeds_probe_not_used_for_fit'})
    graph = defaultdict(list)
    for t in transitions:
        graph[t['state_id']].append(t)
    for sid, edges in graph.items():
        snapshot = before_snapshot(rows[sid])
        for t in edges:
            t['query']['visible_candidates'] = len(edges)
            t['query'] = g.build_query(t['query'], snapshot)
            t['g_estimates'] = {name: g.estimate(m, t['query']) for name,m in models.items()}
            assert t['body_input_id'] not in snapshot['used_input_ids']
    write(out/'transitions.json', transitions)
    start = next(sid for sid,s in states.items() if s['read_count'] == 0 and s['title_count'] == 3)
    policy_runs = []
    for budget in (1, 2, 3):
        for cost in (0., .001, .01):
            for name in ('stop', 'fixed', 'random', 'baseline', 'combined'):
                policy_runs.append({**rollout(graph, start, policy='g' if name in models else name,
                    budget=budget, cost=cost,
                    scorer=(lambda a, model=models.get(name): g.estimate(model, a['query']))),
                    'comparison_policy': name})
    for run in policy_runs:
        for path in run['paths']:
            exact = gain.signed_gain(predictions[start], predictions[path['end_state_id']], outcome)
            assert abs(exact-path['gain']) < 1e-12
    write(out/'policy-replays.json', policy_runs)
    for path,digest in bindings.items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest
    report = {'physical_match_id': MATCH, 'new_nonleeds_matches_for_current_g_evaluation': 1,
        'source_articles': len(docs), 'source_assertions': sum(len(r['response']['claims']) for r in records),
        'states': len(states), 'transitions': len(transitions),
        'gains': {'positive': sum(t['signed_gain'] > 1e-12 for t in transitions),
                  'negative': sum(t['signed_gain'] < -1e-12 for t in transitions),
                  'zero': sum(abs(t['signed_gain']) <= 1e-12 for t in transitions)},
        'title_only_prediction': predictions[start], 'no_text_prior': prior['probabilities'],
        'initial_actions': [{k: t[k] for k in ('body_url','prediction_before','prediction_after','signed_gain','g_estimates')}
                            for t in graph[start]],
        'g_mse': {name: float(np.mean([(t['g_estimates'][name]-t['signed_gain'])**2 for t in transitions]))
                  for name in models},
        'zero_mse': float(np.mean([t['signed_gain']**2 for t in transitions])),
        'f_fit_ids': labeler['head']['head']['encoder']['fit_match_ids'],
        'g_fit_ids': models['baseline']['fit_match_ids'], 'f_g_fit_calls': 0,
        'base_prior_reconstruction_fits': 1, 'provider_api_calls': 0, 'evaluation_http': 0,
        'production_ready': False, 'historical_version_unknown': True,
        'limitations': ['One match does not establish generalisation or conditional expected value.',
            'Currently fetched historical bodies have unknown historical versions.',
            'Publication-ordered menus are reconstructed, not actual query timestamps or propensities.',
            'The existing resolver retains unavailable-to-available disagreement; no f rule changed for this probe.',
            'Search exposed an outcome snippet; not analyst-blind.',
            'No individual importance weights or verified lineup player counts; collective selection assertions are explicit.']}
    write(out/'report.json', report)
    import json
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
