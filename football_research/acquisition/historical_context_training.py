"""Small historical adapter for the existing role f / signed-ridge g kernels.

Archived, genuinely published articles can be revealed in a retrospective
panel without pretending that an Agent fetched them before kickoff.  Their
capture clocks remain separate.  This adapter deliberately does not change
the stronger prospective registries, frozen loaders, or support contracts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer

from .context_predictor import NUMERIC, PARAMETERS as F_PARAMETERS, _digest, _grams
from .context_value import PARAMETERS as G_PARAMETERS, UNITS, _bounded
from .fixture_roles import ROLES, _mentions
from .role_context_predictor import RoleContextPredictor
from .role_context_value import RoleContextValueModel
from .state import parse_time
from .text_returns import _p

VERSION = 'historical-role-context-adapter-v1'
SUPPORT = {'minimum_A_matches': 3, 'minimum_A_classes': 2, 'minimum_B_matches': 2,
           'minimum_local_matches': 2, 'reference_local_matches': 12,
           'reference_local_dates': 2, 'text_cosine_floor': .2,
           'empirical_base_pseudocount_per_class': .5}
G_SELECTION_VERSION = 'b_group_loo_v1'
# Declared before any C/live inspection by this selection procedure. Twenty is
# the original fixed setting; the rest is a logarithmic grid on average loss.
G_SELECTION_GRID = (None, 20., 10., 1., .1, .01, .001, .0001)
LEGACY_TEXT_PROJECTION = 'legacy_all_observed_links'


def _f_parameters(metadata):
    selection = metadata.get('f_selection')
    if selection is None:
        return F_PARAMETERS
    from .historical_predictor_selection import VERSION as SELECTION_VERSION, REGULARIZATIONS
    if (selection.get('schema') != SELECTION_VERSION or selection.get('status') != 'fitted'
            or selection.get('regularization_grid') != list(REGULARIZATIONS)
            or selection.get('selected_regularization') not in (None, *REGULARIZATIONS)
            or selection.get('base_only_won') is not (selection.get('selected_regularization') is None)):
        raise ValueError('Unsupported frozen A-only predictor selection protocol')
    return {**F_PARAMETERS, 'regularization': selection['selected_regularization'],
            'fit_scheme': 'a_group_loo_v1', 'base_only': selection['base_only_won']}


def _g_parameters(metadata):
    selection = metadata.get('g_selection')
    if selection is None:
        return G_PARAMETERS
    if (selection.get('schema') != G_SELECTION_VERSION
            or selection.get('candidate_regularizations') != list(G_SELECTION_GRID)
            or selection.get('selected_regularization') not in G_SELECTION_GRID):
        raise ValueError('Unsupported frozen B-only selection protocol')
    return {**G_PARAMETERS, 'regularization': selection['selected_regularization'],
            'fit_scheme': G_SELECTION_VERSION,
            'constant_model': selection['selected_regularization'] is None}


def _group_weights(rows):
    counts = Counter(row['physical_match_id'] for row in rows)
    if not counts:
        raise ValueError('Independent B fixtures are required')
    return np.asarray([1./(len(counts)*counts[row['physical_match_id']]) for row in rows])


def _gain_design(rows):
    """Original B encoder recipe, refitted on each fold's training games only."""
    grouped = defaultdict(set)
    for row in rows:
        grouped[row['physical_match_id']].update(row['context']['texts']+row['context']['labels'])
    corpus = [sorted(grouped[mid]) for mid in sorted(grouped)]
    vocabulary, idf = [], []
    if any(any(_grams(text)) for text in corpus):
        encoder = TfidfVectorizer(analyzer=_grams, max_features=G_PARAMETERS['max_features'], sublinear_tf=True)
        encoder.fit(corpus)
        vocabulary, idf = encoder.get_feature_names_out().tolist(), encoder.idf_.tolist()
    model = RoleContextValueModel()
    model.encoder = {'vocabulary': vocabulary, 'idf': idf,
        'numeric_size': len(rows[0]['context']['numeric']),
        'channels': sorted({row['context']['channel'] for row in rows}),
        'sources': sorted({source for row in rows for source in row['context']['sources']})}
    vectors = [model._vector(row['context']) for row in rows]
    model.support_records = [{'physical_match_id': row['physical_match_id'], 'date': row['context']['date'],
        'channel': row['context']['channel'], 'mask': row['context']['mask'], 'lexical': vector[1].tolist()}
        for row, vector in zip(rows, vectors)]
    return model, np.asarray([vector[0] for vector in vectors])


def _weighted_gain_ridge(x, y, weights, regularization):
    """Average-loss ridge, unpenalized intercept; small sample-space solve."""
    x, y, weights = map(np.asarray, (x, y, weights))
    if (x.ndim != 2 or len(x) != len(y) or len(y) != len(weights)
            or not np.allclose(x[:, 0], 1.) or not np.isclose(weights.sum(), 1.)
            or np.any(weights <= 0) or not all(np.all(np.isfinite(a)) for a in (x, y, weights))):
        raise ValueError('Finite average-loss rows with an intercept and positive normalized weights required')
    if regularization is not None and (not math.isfinite(regularization) or regularization <= 0):
        raise ValueError('Ridge strength must be strictly positive; None denotes the constant baseline')
    mean_x, mean_y = weights@x[:, 1:], float(weights@y)
    z = (x[:, 1:]-mean_x)*np.sqrt(weights[:, None])
    gram = z@z.T
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.)
    coefficients = np.zeros(x.shape[1])
    if regularization is not None:
        dual = np.linalg.solve(gram+regularization*np.eye(len(y)), (y-mean_y)*np.sqrt(weights))
        coefficients[1:] = z.T@dual
    coefficients[0] = mean_y-mean_x@coefficients[1:]
    residual_loss = float(weights@((y-x@coefficients)**2))
    penalty = 0. if regularization is None else float(regularization*(coefficients[1:]@coefficients[1:]))
    diagnostics = {'regularization': regularization, 'constant_model': regularization is None,
        'weight_sum': float(weights.sum()), 'weighted_mean_target': mean_y,
        'training_mse': residual_loss, 'objective': residual_loss+penalty,
        'maximum_centered_gram_eigenvalue': float(eigenvalues.max()),
        'effective_nonintercept_degrees_of_freedom': 0. if regularization is None
            else float(np.sum(eigenvalues/(eigenvalues+regularization))),
        'maximum_centered_signal_retention': 0. if regularization is None
            else float(eigenvalues.max()/(eigenvalues.max()+regularization))}
    return coefficients, diagnostics


def _fit_b_group_loo(rows):
    """Select only using B; whole matches, including their text, leave each fold."""
    ids = sorted({row['physical_match_id'] for row in rows})
    if len(ids) < 2:
        raise ValueError('B leave-one-fixture-out selection needs two independent matches')
    scores = [[] for _ in G_SELECTION_GRID]
    folds = []
    for held_out in ids:
        train = [row for row in rows if row['physical_match_id'] != held_out]
        valid = [row for row in rows if row['physical_match_id'] == held_out]
        encoder, x = _gain_design(train)
        y = np.asarray([row['gain'] for row in train])
        xv = np.asarray([encoder._vector(row['context'])[0] for row in valid])
        yv = np.asarray([row['gain'] for row in valid])
        hypotheses = []
        for index, strength in enumerate(G_SELECTION_GRID):
            coefficients, _ = _weighted_gain_ridge(x, y, _group_weights(train), strength)
            error = float(np.mean((yv-xv@coefficients)**2))
            scores[index].append(error)
            hypotheses.append({'regularization': strength, 'validation_mse': error})
        folds.append({'held_out_physical_match_id': held_out,
            'training_physical_match_ids': [mid for mid in ids if mid != held_out],
            'training_transitions': len(train), 'validation_transitions': len(valid),
            'training_encoder_sha256': _digest(encoder.encoder), 'hypotheses': hypotheses})
    means = [float(np.mean(values)) for values in scores]
    winner = 0
    for index, score in enumerate(means[1:], 1):
        # A tiny numerical tie does not justify departing from the simpler fit.
        if score < means[winner]-(1e-15+1e-12*abs(means[winner])):
            winner = index
    model, design = _gain_design(rows)
    coefficients, fit = _weighted_gain_ridge(design, np.asarray([row['gain'] for row in rows]),
                                             _group_weights(rows), G_SELECTION_GRID[winner])
    model.coefficients = coefficients.tolist()
    selection = {'schema': G_SELECTION_VERSION,
        'candidate_regularizations': list(G_SELECTION_GRID),
        'criterion': 'mean_B_held_out_physical_match_signed_gain_squared_error',
        'objective_scale': 'sum_weights_one_equal_physical_fixture_then_equal_transition',
        'fold_encoder_scope': 'training_fixtures_only_including_vocabulary_idf_and_source_categories',
        'tie_rule': 'constant_then_greater_regularization_with_1e-15_plus_1e-12_relative_tolerance',
        'folds': folds, 'mean_validation_mse': means,
        'selected_index': winner, 'selected_regularization': G_SELECTION_GRID[winner],
        'constant_baseline_won': winner == 0, 'final_fit': fit,
        'selection_physical_match_ids': ids, 'selection_rows_sha256': _digest(rows),
        'selection_inputs': 'B_only_no_C_or_live_or_positive_gain_counts',
        'fold_direction': 'group_leave_one_out_not_prospective_rolling_origin',
        'uncertainty': 'not_calibrated_selection_has_only_independent_B_match_count'}
    return model, selection


def _host(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        raise ValueError('An actual HTTP source URL is required')
    return parsed.hostname.lower()


def _fixture(value):
    keys = ('physical_match_id', 'home_team_id', 'away_team_id', 'home_name',
            'away_name', 'competition_id', 'kickoff')
    if not isinstance(value, dict) or any(not isinstance(value.get(k), str) or not value[k].strip() for k in keys):
        raise ValueError('Explicit physical fixture, ordered team identities and names required')
    if value['home_team_id'] == value['away_team_id']:
        raise ValueError('Home and away identities must differ')
    result = {key: value[key] for key in keys}
    result['kickoff'] = parse_time(result['kickoff']).isoformat()
    result['match_id'] = value.get('match_id', value['physical_match_id'])
    if value.get('competition') is not None:
        if not isinstance(value['competition'], str) or not value['competition'].strip():
            raise ValueError('Fixture competition must be a nonempty name when supplied')
        result['competition'] = value['competition']
    for role in ('home', 'away'):
        aliases = value.get(role+'_aliases', [])
        if not isinstance(aliases, list) or any(not isinstance(x, str) or not x.strip() for x in aliases):
            raise ValueError('Fixture aliases must be explicit names')
        result[role+'_aliases'] = sorted(set([result[role+'_name'], *aliases]))
    return result


def _publication(doc, *, allow_unknown=False):
    value = doc.get('source_published_at', doc.get('published_at'))
    if value is None:
        if allow_unknown:
            return None
        raise ValueError('Historical text needs a conservative publication upper bound')
    stamp = parse_time(value)
    modified = doc.get('source_modified_at')
    return max(stamp, parse_time(modified)) if modified else stamp


def _document(doc, *, historical, decision_at, allow_unknown_publication=False):
    text = doc.get('text')
    title = doc.get('title', '')
    if not isinstance(text, str) or not text.strip() or not isinstance(title, str):
        raise ValueError('Article text must actually exist; navigation is not a training body')
    _host(doc['url'])
    captured = parse_time(doc['captured_at'])
    publication = _publication(doc, allow_unknown=(allow_unknown_publication and not historical))
    if publication is not None and publication > captured:
        raise ValueError('Publication upper bound cannot follow actual capture')
    if publication is None:
        if doc.get('available_at') is None:
            raise ValueError('Unknown publication requires an explicit operational available_at')
        if doc.get('source_modified_at') is not None and parse_time(doc['source_modified_at']) > captured:
            raise ValueError('Source modification cannot follow actual capture')
    # Historical availability is an explicit assumption, never the real
    # capture clock moved backwards. Operational inputs use real availability.
    available = publication if historical else parse_time(doc.get('available_at', doc['captured_at']))
    if available > decision_at or (not historical and captured > available):
        raise ValueError('The document is not available at this decision')
    return {**deepcopy(doc), 'title': title,
            'source_published_at': publication.isoformat() if publication is not None else None,
            'captured_at': captured.isoformat(), 'modeled_available_at': available.isoformat()}


def _project(state, *, text_projection=LEGACY_TEXT_PROJECTION):
    fixture = _fixture(state['fixture'])
    mode = state.get('information_mode', 'operational')
    if mode not in {'historical_reconstruction', 'operational'}:
        raise ValueError('Explicit historical reconstruction or operational mode required')
    now, cutoff, kickoff = map(parse_time, (state['decision_at'], state['prediction_cutoff'], fixture['kickoff']))
    if not now <= cutoff < kickoff:
        raise ValueError('Decision and cutoff must precede kickoff')
    prior = None if state.get('prior_probabilities') is None else _p(state['prior_probabilities'])
    if prior is not None and not state.get('prior_basis'):
        raise ValueError('A supplied prior requires its actual basis')
    parts = {'body': set(), 'headline': set(), 'link': set()}
    sources, captures, published = set(), [], []
    documents, document_keys = [], set()
    allow_unknown_publication = state.get('allow_unknown_publication_clock', False)
    if type(allow_unknown_publication) is not bool:
        raise ValueError('Explicit operational publication-clock policy required')
    for doc in state.get('documents', []):
        checked = _document(doc, historical=mode == 'historical_reconstruction', decision_at=now,
                            allow_unknown_publication=allow_unknown_publication)
        key = _digest([checked['url'], checked['title'], checked['text']])
        if key not in document_keys:
            document_keys.add(key)
            documents.append(checked)
    for doc in documents:
        paragraphs = doc.get('paragraphs') or [p for p in doc['text'].splitlines() if p.strip()]
        if (not isinstance(paragraphs, list) or any(not isinstance(p, str) or p not in doc['text'] for p in paragraphs)):
            raise ValueError('Role paragraphs must be actual visible body substrings')
        parts['body'].update(p for p in paragraphs if p.strip())
        if doc['title'].strip():
            parts['headline'].add(doc['title'])
        sources.add(_host(doc['url']))
        captures.append((now-parse_time(doc['modeled_available_at'])).total_seconds()/86400)
        if doc.get('source_published_at') is not None:
            published.append((now-parse_time(doc['source_published_at'])).total_seconds()/86400)
    links = []
    visible_links = state.get('observed_links', [])
    if text_projection != LEGACY_TEXT_PROJECTION:
        from .news_link_projection import VERSION as NEWS_LINK_VERSION, project_news_links
        if text_projection != NEWS_LINK_VERSION:
            raise ValueError('Unsupported historical text projection')
        visible_links, _ = project_news_links(visible_links)
    for link in visible_links:
        if not isinstance(link.get('title'), str) or not link['title'].strip():
            raise ValueError('An observed candidate title must be actual nonempty text')
        if link.get('title_basis') not in {'observed_parent_link', 'historical_headline_panel'}:
            raise ValueError('Candidate title availability basis must be explicit')
        if mode == 'operational' and link['title_basis'] != 'observed_parent_link':
            raise ValueError('Operational candidate titles need actual parent observations')
        if parse_time(link['available_at']) > now:
            raise ValueError('Candidate label was not visible before the query')
        _host(link['url'])
        sources.add(_host(link.get('parent_url', link['url'])))
        parts['link'].add(link['title'])
        links.append(deepcopy(link))
    texts = sorted(set().union(*parts.values()))
    aliases = {side: fixture[side+'_aliases'] for side in ('home', 'away')}
    role_texts = {role: [] for role in ROLES}
    for text in texts:
        role_texts[_mentions(text, aliases)[0]].append(text)
    history = state.get('history', [])
    statuses = [item['status'] for item in history]
    fraction = lambda xs, predicate: sum(predicate(x) for x in xs)/max(1, len(xs))
    def age(xs):
        return float(np.mean([_bounded(x) for x in xs])) if xs else 0.
    total = max(1, sum(map(len, parts.values())))
    numeric = [len(parts[k])/total for k in ('body', 'headline', 'link')]
    numeric += [fraction(statuses, lambda s: s == 'FAILED'), fraction(statuses, lambda s: s == 'NO_DATA'),
                fraction(statuses, lambda s: s not in {'OK', 'FAILED', 'NO_DATA'}), 0.]
    for offset in (1, 2):
        status = statuses[-offset] if len(statuses) >= offset else None
        numeric += [float(status == 'FAILED'), float(status == 'NO_DATA'),
                    float(status is not None and status not in {'OK', 'FAILED', 'NO_DATA'})]
    numeric += [0., age(captures), age(published)]
    assert len(numeric) == len(NUMERIC)
    return {'physical_match_id': fixture['physical_match_id'], 'fixture': fixture,
            'decision_at': now.isoformat(), 'prediction_cutoff': cutoff.isoformat(),
            'kickoff_at': kickoff.isoformat(), 'prior': prior, 'texts': texts,
            'role_texts': role_texts, 'sources': sorted(sources), 'numeric': numeric,
            'observed_links': links, 'documents': documents, 'information_mode': mode}


def _state_for(case, count=0, *, acquired_indices=None, discovered_indices=None):
    docs = case['documents']
    acquired = list(range(count)) if acquired_indices is None else sorted(set(acquired_indices))
    discovered = list(range(len(docs))) if discovered_indices is None else sorted(set(discovered_indices))
    if any(type(i) is not int or not 0 <= i < len(docs) for i in acquired):
        raise ValueError('Revealed articles must belong to this actual archived panel')
    if any(type(i) is not int or not 0 <= i < len(docs) for i in discovered) or not set(acquired) <= set(discovered):
        raise ValueError('Read bodies require previously discovered real article titles')
    background = case.get('background_links', [])
    decision = max([_publication(doc) for doc in docs]
                   +[parse_time(link['available_at']) for link in background])
    background_visible = [{'url': link['url'], 'title': link['title'],
        'parent_url': link['parent_url'], 'title_basis': 'historical_headline_panel',
        'available_at': link['available_at']} for link in background]
    return {'fixture': deepcopy(case['fixture']), 'decision_at': decision.isoformat(),
            'prediction_cutoff': decision.isoformat(), 'information_mode': 'historical_reconstruction',
            'prior_probabilities': deepcopy(case.get('prior_probabilities')),
            'prior_basis': case.get('prior_basis'), 'documents': [deepcopy(docs[i]) for i in acquired],
            'observed_links': [{'url': doc['url'], 'title': doc['title'],
                'parent_url': doc.get('parent_url', doc['url']),
                'title_basis': 'historical_headline_panel',
                'available_at': _publication(doc).isoformat()} for i, doc in enumerate(docs) if i in discovered]
                +background_visible,
            'history': [{'action_id': 'article-'+str(i), 'status': 'OK'} for i in acquired]}


def _panel(case):
    """Unique subset/action pairs, never duplicate permutations or fake bodies."""
    width = len(case['documents'])
    if width > 3:
        raise ValueError('This bounded historical panel adapter accepts at most three archived articles per match')
    for mask in range(1 << width):
        acquired = [i for i in range(width) if mask & (1 << i)]
        yield mask, acquired, _state_for(case, acquired_indices=acquired)


def _discovery_panel(case):
    """Each real discovered-title/read-body subset once, including empty start.

    All states share the published panel clock; title discovery is a modeled
    visibility condition, not a claim about an actual historical search.
    """
    width = len(case['documents'])
    if width > 3:
        raise ValueError('This bounded historical panel adapter accepts at most three archived articles per match')
    for discovered_mask in range(1 << width):
        discovered = [i for i in range(width) if discovered_mask & (1 << i)]
        for acquired_mask in range(1 << width):
            if acquired_mask & ~discovered_mask:
                continue
            acquired = [i for i in discovered if acquired_mask & (1 << i)]
            yield discovered_mask, acquired_mask, discovered, acquired, _state_for(
                case, acquired_indices=acquired, discovered_indices=discovered)


def _action(doc, index):
    return {'action_id': 'article-'+str(index), 'tool': 'read_url', 'url': doc['url'],
            'title': doc['title'], 'cost': {'money': 0., 'requests': 1., 'latency_ms': 0., 'tokens': 0.},
            'cost_status': 'simulated'}


def _verified_jsonl_archive(path_value, expected_sha256, *, description):
    """Read an immutable JSONL archive after checking its recorded digest."""
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f'{description} archive path is required')
    path = Path(path_value)
    if not path.exists():
        raise ValueError(f'{description} archive is missing')
    raw = path.read_bytes()
    if not isinstance(expected_sha256, str) or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f'{description} archive bytes differ from the recorded SHA256')
    try:
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{description} archive is not valid JSONL') from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'{description} archive rows must be JSON objects')
    return path, rows


def _prepare_information_stream_label(case, fixture, label, *, result_clock):
    """Bind a footballmind-information-stream case to immutable source ledgers.

    The stream package deliberately omits canonical provider IDs in some rows,
    so the adapter-derived IDs are accepted only after the physical fixture,
    ordered names, competition and kickoff have been checked against the
    archived fixture ledger.  Both the real post-hoc observation clock and the
    reconstructed source-result clock remain in the provenance.
    """
    if result_clock not in {'observed', 'source_reconstruction'}:
        raise ValueError("result_clock must be 'observed' or 'source_reconstruction'")
    _, fixture_rows = _verified_jsonl_archive(label.get('fixtures_path'),
                                              label.get('fixtures_sha256'),
                                              description='Fixture')
    physical_id = fixture['physical_match_id']
    matches = [row for row in fixture_rows if row.get('physical_match_id') == physical_id]
    if len(matches) != 1:
        raise ValueError('One unambiguous archived fixture is required')
    source_fixture = matches[0]
    fixture_kickoff = parse_time(fixture['kickoff'])
    source_kickoff = source_fixture.get('kickoff')
    if (source_fixture.get('home_team') != fixture['home_name']
            or source_fixture.get('away_team') != fixture['away_name']
            or source_fixture.get('competition') != fixture.get('competition')
            or not isinstance(source_kickoff, str)
            or parse_time(source_kickoff) != fixture_kickoff):
        raise ValueError('Stream fixture identity/order/competition/kickoff differs from fixtures.jsonl')

    result_path, result_rows = _verified_jsonl_archive(label.get('path'), label.get('sha256'),
                                                       description='Result')
    label_source = label.get('source') or label.get('source_provider')
    label_id = label.get('source_id')
    rows = [row for row in result_rows
            if str(row.get('source_id')) == str(label_id)
            and (row.get('source') or row.get('source_provider')) == label_source
            and row.get('physical_match_id') == physical_id]
    if len(rows) != 1:
        raise ValueError('One unambiguous stream result row is required')
    row = rows[0]
    raw_result_path = label.get('raw_result_path') or row.get('raw_path')
    raw_result_sha = label.get('raw_result_sha256') or row.get('raw_sha256')
    if not isinstance(raw_result_path, str) or not raw_result_path:
        raise ValueError('Raw result path is required')
    raw_path = Path(raw_result_path)
    if not raw_path.exists() or sha256(raw_path.read_bytes()).hexdigest() != raw_result_sha:
        raise ValueError('Raw result bytes differ from the recorded SHA256')
    hg = row.get('regulation_home_goals', row.get('home_goals'))
    ag = row.get('regulation_away_goals', row.get('away_goals'))
    if (row.get('result_status') not in (None, 'finished')
            or row.get('score_period') != 'regulation'
            or type(hg) is not int or type(ag) is not int or min(hg, ag) < 0):
        raise ValueError('Archived result must be a finished regulation-time score')
    expected_outcome = 0 if hg > ag else 1 if hg == ag else 2
    if case.get('outcome') != expected_outcome:
        raise ValueError('Fixture or outcome differs from its archived regulation result')
    for key in ('home_team_id', 'away_team_id', 'competition_id'):
        value = row.get(key)
        if value is not None and value != fixture[key]:
            raise ValueError('Fixture identity differs from its archived regulation result')
    if row.get('kickoff') is not None and parse_time(row['kickoff']) != fixture_kickoff:
        raise ValueError('Fixture kickoff differs from its archived regulation result')
    observed_raw = row.get('result_observed_at') or row.get('observed_at')
    source_raw = row.get('source_result_available_at') or row.get('result_available_at')
    observed_at = parse_time(observed_raw) if observed_raw else None
    source_at = parse_time(source_raw) if source_raw else None
    if observed_at is None or observed_at < fixture_kickoff:
        raise ValueError('A real post-hoc result observation at or after kickoff is required')
    if source_at is not None and source_at < fixture_kickoff:
        raise ValueError('Reconstructed result availability cannot precede kickoff')
    if result_clock == 'observed':
        result_at = observed_at
    else:
        if source_at is None:
            raise ValueError('Source reconstruction result availability is required')
        result_at = source_at
    provenance = {
        'result_clock': result_clock,
        'result_available_at_basis': 'actual_result_observed_at' if result_clock == 'observed' else 'source_result_available_at_reconstruction',
        'actual_result_observed_at': observed_at.isoformat(),
        'source_result_available_at': source_at.isoformat() if source_at else None,
        'source_result_available_at_basis': 'source_reconstruction' if source_at else None,
        'source_schema': 'footballmind-information-stream/v1',
        'path': str(result_path), 'sha256': label.get('sha256'),
        'source': label_source, 'source_id': label_id,
    }
    return result_at, row, provenance


def prepare_historical_cases(cases, *, result_clock='source_reconstruction'):
    """Validate and normalize immutable historical panels.

    Legacy constructed cases retain their strict original ledger contract.
    The information-stream v1 package follows an explicit branch so missing
    provider IDs are never mistaken for verified equality, and its synthetic
    result clock is visible in the returned provenance.
    """
    if result_clock not in {'observed', 'source_reconstruction'}:
        raise ValueError("result_clock must be 'observed' or 'source_reconstruction'")
    selected, excluded, seen = [], [], set()
    article_matches, label_files = {}, {}
    background_archives, target_background_removed = {}, 0
    for original in cases:
        case = deepcopy(original)
        fixture = _fixture(case['fixture'])
        mid = fixture['physical_match_id']
        if mid in seen:
            raise ValueError('One independent physical fixture must occupy exactly one case')
        seen.add(mid)
        y = case.get('outcome')
        if type(y) is not int or y not in (0, 1, 2):
            raise ValueError('A real regulation-time home/draw/away label is required')
        label = case.get('label_provenance') or {}
        schema = label.get('source_schema')
        if schema == 'footballmind-information-stream/v1':
            result_at, row, provenance = _prepare_information_stream_label(
                case, fixture, label, result_clock=result_clock)
            case['label_provenance'] = {**label, **provenance}
        else:
            # Strict legacy path: every identity/time field is required and
            # must equal the case projection.  This preserves the old tests and
            # prevents the permissive v1 fallbacks from weakening old corpora.
            result_at = parse_time(case['result_available_at'])
            if result_at < parse_time(fixture['kickoff']):
                raise ValueError('Result availability cannot precede kickoff')
            label_path = Path(label['path'])
            label_key = (str(label_path), label.get('sha256'))
            if label_key not in label_files:
                raw = label_path.read_bytes()
                if sha256(raw).hexdigest() != label.get('sha256'):
                    raise ValueError('Result archive bytes differ from the recorded SHA256')
                label_files[label_key] = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
            label_source = label.get('source') or label.get('source_provider')
            rows = [row for row in label_files[label_key]
                    if str(row.get('source_id')) == str(label.get('source_id'))
                    and (row.get('source') or row.get('source_provider')) == label_source]
            if len(rows) != 1:
                raise ValueError('One unambiguous archived result is required')
            row = rows[0]
            hg, ag = row.get('home_goals'), row.get('away_goals')
            if (row.get('score_period') != 'regulation' or type(hg) is not int or type(ag) is not int
                    or min(hg, ag) < 0
                    or y != (0 if hg > ag else 1 if hg == ag else 2)
                    or row.get('home_team_id') != fixture['home_team_id']
                    or row.get('away_team_id') != fixture['away_team_id']
                    or row.get('competition_id') != fixture['competition_id']
                    or not isinstance(row.get('kickoff'), str)
                    or parse_time(row['kickoff']) != parse_time(fixture['kickoff'])
                    or not isinstance(row.get('result_available_at'), str)
                    or parse_time(row['result_available_at']) != result_at):
                raise ValueError('Fixture or outcome differs from its archived regulation result')
            case['label_provenance'] = {**label, 'result_clock': 'legacy_case',
                                        'result_available_at_basis': 'legacy_case_result_available_at'}
        docs, hashes = [], set()
        for raw_doc in case.get('documents', []):
            # The adapter may retain ``body`` until a derived projection is
            # written; normalize it without touching the source archive.
            if not raw_doc.get('text') and isinstance(raw_doc.get('body'), str):
                raw_doc['text'] = raw_doc['body']
            if _publication(raw_doc) >= parse_time(fixture['kickoff']):
                excluded.append({'physical_match_id': mid, 'url': raw_doc.get('url'),
                                 'reason': 'publication_or_modification_not_prematch'})
                continue
            path = Path(raw_doc['raw_path'])
            if not path.exists() or sha256(path.read_bytes()).hexdigest() != raw_doc['raw_sha256']:
                raise ValueError('Archived source bytes differ from the recorded SHA256')
            key = _digest([raw_doc['url'], raw_doc['raw_sha256'], raw_doc.get('text')])
            if key in hashes:
                continue
            article_matches.setdefault(raw_doc['raw_sha256'], set()).add(mid)
            hashes.add(key)
            doc = _document(raw_doc, historical=True, decision_at=_publication(raw_doc))
            if not doc['title'].strip():
                raise ValueError('Historical reveal panels require an actual archived headline')
            docs.append(doc)
        if not docs:
            excluded.append({'physical_match_id': mid, 'reason': 'no_usable_prematch_article'})
            continue
        background, background_seen = [], set()
        document_urls = {doc['url'] for doc in docs}
        for link in case.get('background_links', []):
            if link['url'] in document_urls:
                target_background_removed += 1
                continue
            if not isinstance(link.get('title'), str) or not link['title'].strip():
                raise ValueError('Background titles require real nonempty archived link text')
            _host(link['url']); _host(link['parent_url'])
            publication = parse_time(link['source_published_at'])
            available = parse_time(link['available_at'])
            if available < publication:
                raise ValueError('Background title availability cannot precede its source publication')
            if available >= parse_time(fixture['kickoff']):
                excluded.append({'physical_match_id': mid, 'url': link['url'],
                                 'reason': 'background_title_not_prematch'})
                continue
            if link.get('captured_at') and publication > parse_time(link['captured_at']):
                raise ValueError('Background source publication cannot follow its actual capture')
            binding = (link['raw_path'], link['raw_sha256'])
            if binding not in background_archives:
                path = Path(link['raw_path'])
                if not path.exists() or sha256(path.read_bytes()).hexdigest() != link['raw_sha256']:
                    raise ValueError('Archived background source bytes differ from the recorded SHA256')
                background_archives[binding] = True
            key = (link['url'], link['title'], link['parent_url'])
            if key not in background_seen:
                background_seen.add(key); background.append(deepcopy(link))
        case.update(fixture=fixture, documents=sorted(docs, key=lambda d: (_publication(d), d['url'])),
                    result_available_at=result_at.isoformat(), background_links=background)
        _project(_state_for(case, 0))
        selected.append(case)
    selected.sort(key=lambda case: (parse_time(case['fixture']['kickoff']), case['fixture']['physical_match_id']))
    return selected, {'excluded': excluded, 'physical_matches': len(selected),
                      'archived_articles': sum(len(c['documents']) for c in selected),
                      'background_titles': sum(len(c['background_links']) for c in selected),
                      'target_titles_removed_from_background': target_background_removed,
                      'shared_raw_articles': {k: sorted(v) for k, v in article_matches.items() if len(v) > 1}}


class HistoricalContextModel:
    """One frozen pair; computations are the existing role softmax/ridge kernels."""
    def __init__(self):
        self.f, self.g = RoleContextPredictor(), RoleContextValueModel()
        self.metadata, self.model_version = {}, ''

    def _probabilities(self, state):
        projected = _project(state, text_projection=self.metadata.get('text_projection', LEGACY_TEXT_PROJECTION))
        base = projected['prior'] if projected['prior'] is not None else self.metadata['empirical_base']
        vector, known = self.f._vector(projected)
        base_only = self.metadata.get('f_selection', {}).get('base_only_won') is True
        residual = vector @ np.asarray(self.f.coefficients) if known and not base_only else np.zeros(3)
        probabilities = softmax(np.log(base)+residual).tolist() if known and not base_only else list(base)
        return projected, probabilities, known, list(base)

    def _heldout(self, projected, *, value=False):
        mid = projected['physical_match_id']
        forbidden = self.metadata['A_match_ids'] + (self.metadata['B_match_ids'] if value else [])
        if mid in forbidden:
            raise ValueError('Evaluation cannot reuse the relevant f/g training fixture')
        cutoff = self.metadata['g_training_as_of'] if value else self.metadata['f_training_as_of']
        if parse_time(projected['decision_at']) <= parse_time(cutoff):
            raise ValueError('The query must follow the frozen model information cutoff')
        if projected['information_mode'] == 'operational' and parse_time(projected['decision_at']) < parse_time(self.metadata['fit_execution_at']):
            raise ValueError('A live query cannot precede actual fitting execution')

    def predict(self, state):
        self._check()
        projected, p, known, base = self._probabilities(state)
        self._heldout(projected)
        domain = 'missing_prior' if projected['prior'] is None else 'provided_prior'
        result = {'probabilities': p, 'base_probabilities': base, 'model_version': self.model_version,
                'physical_match_id': projected['physical_match_id'], 'prior_domain': domain,
                'known_character_features': known, 'fallback_reason': None if known else 'no_known_visible_text',
                'support_status': 'prototype_text_projection' if domain in self.metadata['A_prior_domains'] else 'prior_domain_extrapolation',
                'calibration_status': 'not_established', 'evidence_origin': 'historical_reconstruction',
                'prediction_kind': 'same_role_offset_softmax', 'production_ready': False}
        if 'f_selection' in self.metadata:
            selection = self.metadata['f_selection']
            result['predictor_selection'] = {key: deepcopy(selection[key]) for key in
                ('schema', 'base_only_won', 'selected_regularization', 'physical_matches', 'selection_scope')}
            result['context_signal_enabled'] = not selection['base_only_won']
            if selection['base_only_won']:
                result.update(prediction_kind='A_selected_empirical_base_only',
                    fallback_reason='A_group_validation_selected_no_text_effect')
        return result

    def _context(self, state, action):
        projected, p, _, base = self._probabilities(state)
        if action.get('tool') != 'read_url':
            raise ValueError('This fitted corpus contains article reading, not a search response model')
        host = _host(action['url'])
        links = [link for link in projected['observed_links'] if link['url'] == action['url']]
        if not links or (action.get('title') is not None and action['title'] not in {link['title'] for link in links}):
            raise ValueError('Candidate labels must come from actually visible parent links/panel')
        labels = sorted({link['title'] for link in links})
        aliases = {r: projected['fixture'][r+'_aliases'] for r in ('home', 'away')}
        role_labels = {role: [] for role in ROLES}
        for text in labels:
            role_labels[_mentions(text, aliases)[0]].append(text)
        cost = action.get('cost', {})
        if action.get('cost_status') not in {'known', 'estimated', 'simulated'}:
            raise ValueError('Physical query costs must have a declared interpretation')
        numeric = projected['numeric']+p+base
        numeric += [_bounded((parse_time(projected['kickoff_at'])-parse_time(projected['decision_at'])).total_seconds()/3600)]
        missing = projected['prior'] is None
        numeric += [float(missing)] + [_bounded(cost.get(k, 0)) for k in UNITS]
        return {'texts': projected['texts'], 'labels': labels, 'numeric': numeric,
                'sources': projected['sources'], 'channel': 'read_url|'+host,
                'mask': [bool(projected['texts']), bool(labels), missing]
                    +[bool(projected['role_texts'][r]) for r in ROLES]+[bool(role_labels[r]) for r in ROLES],
                'role_texts': projected['role_texts'], 'candidate_role_texts': role_labels,
                'prior_domain': 'missing_prior' if missing else 'provided_prior',
                'date': parse_time(projected['kickoff_at']).date().isoformat(),
                'decision_at': projected['decision_at'], 'kickoff_at': projected['kickoff_at']}

    def estimate(self, state, action, *, state_generalization=False):
        """Strict local support by default; optional uncalibrated state-pattern extrapolation.

        The research option changes no learned parameters and keeps candidate
        roles, source/domain checks and independent-neighbor requirements.
        """
        if type(state_generalization) is not bool:
            raise ValueError('State-pattern generalization requires an explicit boolean research option')
        self._check()
        projected = _project(state, text_projection=self.metadata.get('text_projection', LEGACY_TEXT_PROJECTION))
        self._heldout(projected, value=True)
        if action.get('tool') != 'read_url':
            return self._unknown('unmodeled_tool')
        if any(doc['url'] == action['url'] for doc in projected['documents']):
            return self._unknown('already_read_document')
        context = self._context(state, action)
        vector, lexical = self.g._vector(context)
        nearby, exact_nearby = [], []
        reason = None
        if context['prior_domain'] not in self.metadata['B_prior_domains']:
            reason = 'unseen_prior_domain'
        elif context['channel'] not in self.g.encoder['channels']:
            reason = 'unseen_tool_or_request_source'
        elif set(context['sources'])-set(self.g.encoder['sources']):
            reason = 'unseen_visible_source'
        else:
            for row in self.g.support_records:
                if row['channel'] != context['channel']:
                    continue
                exact_state = row['mask'] == context['mask']
                if state_generalization:
                    candidate_offset = 3+len(ROLES)
                    if row['mask'][:3] != context['mask'][:3] or row['mask'][candidate_offset:] != context['mask'][candidate_offset:]:
                        continue
                elif not exact_state:
                    continue
                old = np.asarray(row['lexical'])
                norm = np.linalg.norm(lexical)*np.linalg.norm(old)
                similarity = float(lexical@old/norm) if norm else 0.
                if similarity >= SUPPORT['text_cosine_floor']:
                    nearby.append(row)
                    if exact_state:
                        exact_nearby.append(row)
            if len({r['physical_match_id'] for r in nearby}) < SUPPORT['minimum_local_matches']:
                reason = 'insufficient_independent_local_support'
        support = {'physical_matches': len({r['physical_match_id'] for r in nearby}),
                   'kickoff_dates': len({r['date'] for r in nearby}), 'channel': context['channel'],
                   'minimum_matches': SUPPORT['minimum_local_matches'], 'text_cosine_floor': SUPPORT['text_cosine_floor']}
        extrapolation = False
        if state_generalization:
            exact_matches = len({r['physical_match_id'] for r in exact_nearby})
            extrapolation = (reason is None and exact_matches < SUPPORT['minimum_local_matches'])
            support.update(exact_state_matches=exact_matches, state_pattern_extrapolation=extrapolation)
        if reason:
            return {**self._unknown(reason), 'support': support}
        reference_rows = exact_nearby if state_generalization else nearby
        enough = (len({r['physical_match_id'] for r in reference_rows}) >= SUPPORT['reference_local_matches']
                  and len({r['date'] for r in reference_rows}) >= SUPPORT['reference_local_dates'])
        gain = float(vector@np.asarray(self.g.coefficients))
        return {'expected_gain': gain, 'expected_signed_gain': gain,
                'support_status': 'prototype_state_generalization' if extrapolation else 'supported' if enough else 'prototype_supported',
                'reason': 'uncalibrated_state_pattern_generalization' if extrapolation else 'empirical_context' if enough else 'small_sample_empirical_context',
                'support': support, 'model_version': self.model_version,
                'value_kind': 'empirical_signed_fixed_predictor_gain_not_Bayes_EVSI',
                'cost_deducted': False, 'calibration_status': 'not_established',
                'uncertainty': None, 'stop_certificate': None, 'production_ready': False}

    def _unknown(self, reason):
        return {'expected_gain': None, 'expected_signed_gain': None, 'support_status': 'UNKNOWN',
                'reason': reason, 'model_version': self.model_version, 'cost_deducted': False,
                'value_kind': 'empirical_signed_fixed_predictor_gain_not_Bayes_EVSI',
                'calibration_status': 'not_established', 'uncertainty': None, 'stop_certificate': None}

    def _payload(self):
        return {'schema': VERSION, 'f_parameters': _f_parameters(self.metadata), 'g_parameters': _g_parameters(self.metadata),
                'prototype_support': SUPPORT, 'metadata': self.metadata,
                'f': {'encoder': self.f.encoder, 'coefficients': self.f.coefficients},
                'g': {'encoder': self.g.encoder, 'coefficients': self.g.coefficients,
                      'support_records': self.g.support_records}}

    def _check(self):
        if self.model_version != VERSION+':'+_digest(self._payload())[:24]:
            raise ValueError('Frozen historical f/g artifact changed')

    def save(self, folder):
        self._check()
        path = Path(folder)/'model.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump({**self._payload(), 'model_version': self.model_version}, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return path

    @classmethod
    def load(cls, folder):
        path = Path(folder)
        payload = json.loads((path/'model.json' if path.is_dir() else path).read_text(encoding='utf-8'))
        if payload.get('schema') != VERSION or payload.get('prototype_support') != SUPPORT or payload.get('f_parameters') != _f_parameters(payload.get('metadata', {})) or payload.get('g_parameters') != _g_parameters(payload.get('metadata', {})):
            raise ValueError('Unsupported historical kernel or support settings')
        model = cls()
        model.metadata, model.model_version = payload['metadata'], payload['model_version']
        for key in ('encoder', 'coefficients'):
            setattr(model.f, key, payload['f'][key])
            setattr(model.g, key, payload['g'][key])
        model.g.support_records = payload['g']['support_records']
        model._check()
        return model


def fit_historical_context(cases, output_dir=None, *, dataset_version, a_fraction=.6, b_fraction=.2,
                           maximum_panel_articles=3, g_fit_scheme='fixed_legacy',
                           text_projection=LEGACY_TEXT_PROJECTION, f_fit_scheme='fixed_legacy'):
    """Fit f on A, recompute true-body signed gains on B, and inspect C.

    Every unique discovered-title/body-subset/action of at most three genuinely
    archived articles is enumerated, without an outcome-selected order. Only
    discovered headlines enter a state; unread bodies remain hidden. These are
    counterfactual reveals of real content, not historical Agent trajectories
    or invented on-date fetches. An empty startup state is included for f.
    """
    if not dataset_version or not 0 < a_fraction < 1 or not 0 < b_fraction < 1 or a_fraction+b_fraction >= 1:
        raise ValueError('An explicit dataset and nonempty chronological A/B/C proportions are required')
    if type(maximum_panel_articles) is not int or not 1 <= maximum_panel_articles <= 3:
        raise ValueError('The bounded real panel has between one and three articles')
    if g_fit_scheme not in {'fixed_legacy', G_SELECTION_VERSION}:
        raise ValueError('Unknown explicit historical gain fitting scheme')
    if f_fit_scheme not in {'fixed_legacy', 'a_group_loo_v1'}:
        raise ValueError('Unknown explicit historical predictor fitting scheme')
    if text_projection != LEGACY_TEXT_PROJECTION:
        from .news_link_projection import VERSION as NEWS_LINK_VERSION
        if text_projection != NEWS_LINK_VERSION:
            raise ValueError('Unknown explicit historical text projection')
    cases, audit = prepare_historical_cases(cases)
    n = len(cases)
    na, nb = max(1, int(n*a_fraction)), max(1, int(n*b_fraction))
    groups = {'A': cases[:na], 'B': cases[na:na+nb], 'C': cases[na+nb:]}
    report = {'schema': VERSION, 'status': 'not_ready', 'dataset_version': dataset_version,
              'evidence_origin': 'historical_reconstruction', 'audit': audit,
              'splits': {k: [r['fixture']['physical_match_id'] for r in v] for k, v in groups.items()},
              'reasons': [], 'scope': 'real_archived_text_reconstructed_reveal_panels_not_prospective_Agent_trajectories'}
    A, B, C = [groups[k] for k in ('A', 'B', 'C')]
    counts = Counter(c['outcome'] for c in A)
    if len(A) < SUPPORT['minimum_A_matches'] or len(counts) < SUPPORT['minimum_A_classes']:
        report['reasons'].append('insufficient_independent_A_matches_or_classes')
    if len(B) < SUPPORT['minimum_B_matches'] or not C:
        report['reasons'].append('insufficient_independent_B_or_C_matches')
    if any(len(c['documents']) > maximum_panel_articles for c in cases):
        report['reasons'].append('historical_panel_exceeds_declared_article_scope')
    # Since early pre-match articles can overlap in time with earlier matches,
    # purging uses the actual reconstructed decision, not just kickoff order.
    f_cutoff = max((parse_time(c['result_available_at']) for c in A), default=None)
    g_cutoff = max((parse_time(c['result_available_at']) for c in B), default=None)
    if f_cutoff and any(parse_time(_state_for(c, 0)['decision_at']) <= f_cutoff for c in B+C):
        report['reasons'].append('A_labels_overlap_B_or_C_prediction_time')
    if g_cutoff and any(parse_time(_state_for(c, 0)['decision_at']) <= g_cutoff for c in C):
        report['reasons'].append('B_labels_overlap_C_prediction_time')
    ownership = {}
    for group, rows in groups.items():
        for case in rows:
            for doc in case['documents']:
                ownership.setdefault(doc['raw_sha256'], set()).add(group)
    if any(len(owner) > 1 for owner in ownership.values()):
        report['reasons'].append('same_raw_article_crosses_training_partitions')
    if report['reasons']:
        return None, report
    model = HistoricalContextModel()
    pseudo = SUPPORT['empirical_base_pseudocount_per_class']
    base = [(counts[y]+pseudo)/(len(A)+3*pseudo) for y in range(3)]
    model.metadata = {'dataset_version': dataset_version, 'fit_execution_at': datetime.now(timezone.utc).isoformat(),
        'f_training_as_of': f_cutoff.isoformat(), 'g_training_as_of': g_cutoff.isoformat(),
        'A_match_ids': report['splits']['A'], 'B_match_ids': report['splits']['B'],
        'C_match_ids': report['splits']['C'], 'empirical_base': base,
        'empirical_base_counts': {str(y): counts[y] for y in range(3)},
        'A_prior_domains': sorted({'missing_prior' if c.get('prior_probabilities') is None else 'provided_prior' for c in A}),
        'B_prior_domains': sorted({'missing_prior' if c.get('prior_probabilities') is None else 'provided_prior' for c in B}),
        'evidence_origin': 'historical_reconstruction', 'capture_times_rewritten': False,
        'reveal_semantics': 'each_unique_discovered_title_subset_D_read_body_subset_R_and_unread_action_a_in_D_minus_R',
        'counterfactual_scope': 'real_archived_article_panel_not_historical_Agent_action_trajectories',
        'maximum_panel_articles': maximum_panel_articles, 'duplicate_permutations_counted': False,
        'undiscovered_titles_in_state': False, 'empty_discovered_panel_in_f': True,
        'empty_startup_in_f': not any(c.get('background_links') for c in A),
        'background_title_semantics': 'actual_archived_prematch_menu_titles_always_visible_not_readable_body_actions',
        'decision_clock': 'common_latest_actual_panel_publication_or_background_availability_no_claim_of_earlier_discovery',
        'availability_assumption': 'source_publication_or_modification_upper_bound_not_actual_capture',
        'training_weighting': 'equal_physical_fixture_then_equal_prefix_or_transition',
        'kernel_reuse': ['RoleContextPredictor._fit_rows/_vector', 'RoleContextValueModel._fit_regression/_vector'],
        'calibration_status': 'not_established', 'production_ready': False,
        'corpus_sha256': _digest(cases), 'source_bindings': sorted({(d['raw_path'], d['raw_sha256'])
            for c in cases for d in c['documents']+c.get('background_links', [])})}
    if text_projection != LEGACY_TEXT_PROJECTION:
        model.metadata['text_projection'] = text_projection
    f_rows = []
    for case in A:
        for _, _, _, _, visible in _discovery_panel(case):
            state = _project(visible, text_projection=text_projection)
            if f_fit_scheme == 'fixed_legacy':
                state['prior'] = state['prior'] if state['prior'] is not None else list(base)
            f_rows.append({'physical_match_id': case['fixture']['physical_match_id'], 'outcome': case['outcome'], 'state': state})
    if f_fit_scheme == 'a_group_loo_v1':
        from .historical_predictor_selection import fit_A_predictor
        predictor, selection, selected_base = fit_A_predictor(f_rows, pseudocount=pseudo)
        report['f_selection'] = deepcopy(selection)
        if predictor is None:
            report['reasons'].append('A_predictor_selection_'+selection['status'])
            return None, report
        if selected_base != base:
            raise ValueError('The final A-only base must equal the declared independent-fixture base')
        model.f = predictor
        model.metadata['f_selection'] = selection
        model.metadata['f_optimizer'] = deepcopy(selection['final_fit'])
        model.metadata['kernel_reuse'][0] = 'A_selected_same_RoleContextPredictor_vector_and_softmax_objective'
    else:
        _, result = model.f._fit_rows(f_rows)
        model.metadata['f_optimizer'] = {'iterations': int(result.nit), 'objective': float(result.fun)}
    gains = []
    for case in B:
        for discovered_mask, acquired_mask, discovered, acquired, before in _discovery_panel(case):
            for i in discovered:
                if i in acquired:
                    continue
                doc = case['documents'][i]
                after = _state_for(case, acquired_indices=acquired+[i], discovered_indices=discovered)
                p = model._probabilities(before)[1]
                q = model._probabilities(after)[1]
                y, mid = case['outcome'], case['fixture']['physical_match_id']
                gain = math.log(q[y])-math.log(p[y])
                gains.append({'sample_id': mid+':'+str(discovered_mask)+':'+str(acquired_mask)+':'+str(i),
                              'physical_match_id': mid, 'outcome': y,
                              'gain': gain, 'prediction_before': p, 'prediction_after': q,
                              'discovered_indices': discovered, 'acquired_indices': acquired, 'action_index': i,
                              'url': doc['url'], 'context': model._context(before, _action(doc, i))})
    if g_fit_scheme == G_SELECTION_VERSION:
        model.g, selection = _fit_b_group_loo(gains)
        model.metadata['g_selection'] = selection
        objective = selection['final_fit']['objective']
        report['g_selection'] = deepcopy(selection)
    else:
        _, objective = model.g._fit_regression(gains)
    model.metadata['g_objective'] = objective
    model.metadata['B_targets_sha256'] = _digest(gains)
    if f_fit_scheme != 'fixed_legacy':
        model.metadata['fit_started_at'] = model.metadata['fit_execution_at']
        model.metadata['fit_execution_at'] = datetime.now(timezone.utc).isoformat()
    model.model_version = VERSION+':'+_digest(model._payload())[:24]
    checks = []
    for case in C:
        for _, _, discovered, acquired, before in _discovery_panel(case):
            for i in discovered:
                if i in acquired:
                    continue
                doc = case['documents'][i]
                after = _state_for(case, acquired_indices=acquired+[i], discovered_indices=discovered)
                p, q = model.predict(before)['probabilities'], model.predict(after)['probabilities']
                estimate = model.estimate(before, _action(doc, i))
                y = case['outcome']
                checks.append({'physical_match_id': case['fixture']['physical_match_id'], 'url': doc['url'],
                    'outcome': y, 'discovered_indices': discovered, 'acquired_indices': acquired, 'action_index': i,
                    'prediction_before': p, 'prediction_after': q,
                    'actual_signed_gain': math.log(q[y])-math.log(p[y]), 'estimate': estimate})
    signs = Counter('positive' if r['gain'] > 1e-12 else 'negative' if r['gain'] < -1e-12 else 'near_zero' for r in gains)
    report.update(status='fitted_research_prototype', model_version=model.model_version,
                  A_prefixes=len(f_rows), B_transitions=len(gains), B_gain_signs=dict(signs),
                  C_transitions=len(checks), C_computed_estimates=sum(r['estimate']['expected_gain'] is not None for r in checks),
                  C_mean_actual_gain=float(np.mean([r['actual_signed_gain'] for r in checks])),
                  C_check=checks, no_realworld_benefit_claim=True,
                  limits=['small_independent_sample', 'historically_reconstructed_menu_and_availability',
                          'current_archive_may_include_unobserved_historical_revisions',
                          'one_fixed_predictor_no_Bayes_EVSI_or_probability_calibration_guarantee'])
    if output_dir is not None:
        folder = Path(output_dir)
        model.save(folder)
        for name, data in (('report.json', report), ('B-targets.json', gains)):
            with (folder/name).open('x', encoding='utf-8') as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return model, report


__all__ = ['HistoricalContextModel', 'fit_historical_context', 'prepare_historical_cases', 'SUPPORT', 'VERSION']
