"""Versioned research head with explicit semantic support, not information value.

Routing uses frozen reviewed annotations, never outcomes or prediction shifts.
Unsupported claims remain in the audit, with unknown usefulness. Weak context
uses concept features without shared quote vocabulary; selection is distinct
from medical availability. This is a proposed inductive bias, not a learned
relevance classifier or production extractor.
"""
from copy import deepcopy
from . import reviewed_rich_f as rich
from .context_predictor import _digest

VERSION = 'scoped-reviewed-residual-f-v1'
RULE_VERSION = 'reviewed-evidence-routing-v1'
PREDICATES = frozenset({
    'availability:unavailable', 'availability:available',
    'availability:doubtful', 'availability:recovering',
    'selection:rotation_planned', 'selection:unchanged_planned',
    'tactical_plan:attacking', 'tactical_plan:defensive',
    'schedule:congested', 'schedule:rest_advantage',
    'stakes:draw_sufficient', 'stakes:must_win',
})
SELECTION = frozenset({'confirmed_starter', 'confirmed_bench', 'confirmed_squad'})
CONTEXT = frozenset({
    'training_participation', 'partial_training', 'recovery_not_ready',
    'match_decision_pending', 'club_eligibility_pending', 'possible_bench',
    'observed_lineup_change', 'season_ambition', 'match_ambition',
    'season_attacking_ambition', 'match_preparation',
})


def rule_manifest():
    return {'schema': RULE_VERSION, 'predicates': sorted(PREDICATES),
            'selection_concepts': sorted(SELECTION), 'context_concepts': sorted(CONTEXT),
            'unknown_usefulness': None, 'lexical_context_enabled': False,
            'deduplication': 'exact team + atoms + text across title/body; not semantic equivalence',
            'policy': 'reviewer-defined support routes; not a fitted relevance gate'}


def field(atoms, prefix):
    values = [x[len(prefix):] for x in atoms if x.startswith(prefix)]
    if len(values) > 1:
        raise ValueError('Ambiguous reviewed semantic field')
    return values[0] if values else None


def route(row, mode='typed_context'):
    if mode not in ('direct_only', 'typed_context'):
        raise ValueError('Unknown routing mode')
    retained, audit = {}, []
    for e in rich.unique_evidence(row['evidence']):
        if e['channel'] not in rich.CHANNELS or e['team'] not in rich.TEAMS:
            raise ValueError('Unknown evidence role')
        predicate = field(e['atoms'], 'predicate:')
        scope = field(e['atoms'], 'scope:')
        concept = field(e['atoms'], 'concept:')
        lane = 'unsupported'
        if e['team'] != 'unknown' and scope == 'this_match':
            if concept in SELECTION:
                lane = 'selection'
            elif predicate in PREDICATES:
                lane = 'direct'
        if (lane == 'unsupported' and mode == 'typed_context' and
                e['team'] != 'unknown' and scope in ('team_context', 'this_match') and concept in CONTEXT):
            lane = 'context'
        audit.append({'evidence_sha256': _digest(e), 'lane': lane,
                      'concept': concept, 'scope': scope, 'information_value': None})
        if lane == 'unsupported':
            continue
        mapped = deepcopy(e)
        # Exact cross-channel repeats are one assertion, not independent facts.
        mapped['channel'] = 'body'
        if lane in ('selection', 'context'):
            # Do not let a selection imply a medical-clearance feature, or weak
            # context share the lexical feature path of confirmed match facts.
            mapped['atoms'] = [f'lane:{lane}', f'concept:{concept}', f'scope:{scope}']
            mapped['text'] = ''
        else:
            mapped['atoms'] = sorted(set(mapped['atoms'] + ['lane:direct']))
        retained[_digest(mapped)] = mapped
    result = {**deepcopy(row), 'evidence': list(retained.values())}
    return result, audit


def fit(rows, regularization, mode='typed_context'):
    routed = [route(r, mode)[0] for r in rows]
    return {'schema': VERSION, 'mode': mode, 'rules': rule_manifest(),
            'head': rich.fit(routed, regularization), 'production_ready': False,
            'calibration_status': 'not_established', 'g_training': False}


def predict(model, row):
    if model.get('schema') != VERSION or model.get('rules') != rule_manifest():
        raise ValueError('Routing contract differs')
    return rich.predict(model['head'], route(row, model['mode'])[0])


def save(model, path):
    rich.save(model, path)


def load(path):
    import json
    from pathlib import Path
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = value.pop('model_sha256')
    if _digest(value) != digest or value.get('schema') != VERSION or value.get('rules') != rule_manifest():
        raise ValueError('Frozen scoped model binding differs')
    if value.get('mode') not in ('direct_only', 'typed_context') or value['head'].get('schema') != rich.VERSION:
        raise ValueError('Unknown scoped model head')
    return value
