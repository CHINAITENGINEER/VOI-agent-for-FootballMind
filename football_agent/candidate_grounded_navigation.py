'''Evidence-local fixture candidates for autonomous information navigation.

Literal matchup grounding rejects unrelated citations; it is not a semantic
entailment model or world-truth check. Unsupported prose remains an information
need rather than becoming a claimed fixture. Historical protocols stay frozen.
'''
from copy import deepcopy
import re
import unicodedata

from jsonschema import Draft202012Validator

from .audit import digest
from .inquiry_contract import obj
from .target_navigation import TargetNavigationWorkflow, SCHEMA as TARGET_SCHEMA, PROMPT as TARGET_PROMPT

CLUE = {'type': 'string', 'minLength': 1, 'maxLength': 180}
NULLABLE_CLUE = {'anyOf': [{'type': 'null'}, CLUE]}
CANDIDATE = obj({
    'claim_index': {'type': 'integer', 'minimum': 0, 'maximum': 2},
    'identity_quote': {'type': 'string', 'minLength': 8, 'maxLength': 220},
    'team_a_text': CLUE, 'team_b_text': CLUE,
    'connector_text': {'enum': ['VS', 'vs', 'Vs', 'v', 'v.', 'versus', '对阵', '對陣']},
    'date_text': NULLABLE_CLUE, 'time_text': NULLABLE_CLUE, 'scope_text': NULLABLE_CLUE})
SCHEMA = deepcopy(TARGET_SCHEMA)
state = SCHEMA['properties']['target_state']['properties']
state.pop('candidate_label')
state.pop('support_claim_indices')
state['candidate'] = {'anyOf': [{'type': 'null'}, CANDIDATE]}
SCHEMA['properties']['target_state']['required'] = list(state)
PROMPT = TARGET_PROMPT + '''
CANDIDATE GROUNDING OVERRIDE (this version replaces the earlier label/index fields):
target_state has status, candidate, open_requirements. No candidate_label or
support_claim_indices. candidate=null for unresolved. Otherwise use candidate
with claim_index in THIS response, identity_quote contained in that opened-body
claim, and the exact original team_a_text, connector_text, team_b_text.
The identity_quote must contain the literal pair: TEAM_A whitespace CONNECTOR
whitespace TEAM_B, e.g. Gangwon VS Bucheon. Pair order does NOT establish home/away.
This narrow supported syntax is an evidence-admission boundary, not a declaration
that other prose contains no fixture. With ambiguous or unsupported identity,
retain useful claims, candidate=null and fixture_identity open; select useful
information or defer freely. Never manufacture names or connectors to pass it.
At least one original team name must match the task's supplied aliases.
Optional date_text/time_text/scope_text are exact substrings of identity_quote;
unknown is null. Never add a year from publication, retrieval, weekday or memory.
NEXT HOME MATCH is a home-only clue; keep all_fixture_order open. Without an
explicit year in date_text, keep kickoff_date_or_time and reference_time_applicability
open. A quote is a source statement, not independently verified reality.
Prior_information_tests preserve your own scoped observations. An irrelevant
return for fixture order does not make that source useless for all questions.
Choose read/query/defer using unresolved requirements and expected decision change;
no source order is prescribed and no numerical information value is supplied.
'''


class CandidateGroundedProtocol:
    version, schema, prompt = 'candidate-grounded-navigation-v1', SCHEMA, PROMPT


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def grounded_candidate(candidate, claims, contract):
    if candidate is None:
        return None
    index = candidate['claim_index']
    if index >= len(claims):
        raise ValueError('Candidate claim is outside this response')
    quote = candidate['identity_quote']
    parent = claims[index]['quote']
    if parent.count(quote) != 1:
        raise ValueError('Candidate identity must be one exact span in its supporting claim')
    left, right = candidate['team_a_text'], candidate['team_b_text']
    if any(v != v.strip() or not v.strip() for v in (left,right)) or normalized(left) == normalized(right):
        raise ValueError('Candidate requires distinct nonblank team names')
    aliases = {normalized(v) for v in contract['team_aliases']}
    if not ({normalized(left), normalized(right)} & aliases):
        raise ValueError('Candidate belongs to a different task team')
    pattern = r'(?<!\w)' + re.escape(left) + r'\s+' + re.escape(candidate['connector_text']) + r'\s+' + re.escape(right) + r'(?!\w)'
    if len(list(re.finditer(pattern, quote))) != 1:
        raise ValueError('Supporting span does not contain the claimed literal matchup')
    for key in ('date_text', 'time_text', 'scope_text'):
        value = candidate[key]
        if value is not None and (not value.strip() or value != value.strip() or quote.count(value) != 1):
            raise ValueError('Candidate clue is not a unique literal in the same identity span: ' + key)
    blockers = []
    if candidate['date_text'] is None or not re.search(r'(?<![0-9])[12][0-9]{3}(?![0-9])', candidate['date_text']):
        blockers += ['kickoff_date_or_time', 'reference_time_applicability']
    scope = normalized(quote)
    if any(marker in scope for marker in ('next home match', 'next home', '下一主场', '下一主場')):
        blockers.append('all_fixture_order')
    label = left + ' ' + candidate['connector_text'] + ' ' + right
    clues = [candidate[k] for k in ('date_text', 'time_text') if candidate[k]]
    if clues:
        label += ' | ' + ' | '.join(clues)
    return {'candidate_label': label, 'support_claim_indices': [index],
        'required_open_requirements': sorted(set(blockers)),
        'literal_matchup_grounded': True, 'home_away_order_verified': False,
        'semantic_entailment_verified': False, 'world_truth_verified': False}


def project_target_choice(choice, contract):
    Draft202012Validator(SCHEMA).validate(choice)
    projected = deepcopy(choice)
    state = projected['target_state']
    candidate = state.pop('candidate')
    bound = grounded_candidate(candidate, choice['claims'], contract)
    if bound is None:
        if state['status'] != 'unresolved' or 'fixture_identity' not in state['open_requirements']:
            raise ValueError('Ungrounded fixture remains unresolved with an identity information need')
        state.update(candidate_label=None, support_claim_indices=[])
    else:
        if state['status'] == 'unresolved':
            raise ValueError('Unresolved target has no admitted candidate')
        if not set(bound['required_open_requirements']).issubset(state['open_requirements']):
            raise ValueError('Partial date or home-only scope still requires information')
        state.update(candidate_label=bound['candidate_label'],support_claim_indices=bound['support_claim_indices'])
    return projected


class CandidateGroundedWorkflow(TargetNavigationWorkflow):
    version = 'candidate-grounded-navigation-workflow-v1'

    def configuration(self):
        result = super().configuration()
        result.update(protocol=CandidateGroundedProtocol.version,
            protocol_digest=digest({'schema':SCHEMA,'prompt':PROMPT}),
            candidate_grounding='same_span_literal_matchup_not_world_truth')
        return result

    def public(self, run, *, now=None):
        result = super().public(run,now=now)
        feedback = []
        for decision in run['decisions']:
            choice = decision['choice']
            assessment = choice.get('return_assessment')
            if assessment is None:
                continue
            indices = assessment['evidence_claim_indices']
            feedback.append({'assessment':deepcopy(assessment),
                'claims':[deepcopy(choice['claims'][i]) for i in indices],
                'scope':'this_return_for_this_requirement_only_not_source_wide_utility'})
        result['prior_information_tests'] = feedback
        result['candidate_grounding_scope'] = 'literal_matchup_and_clues_only_not_entailment_or_verified_target'
        return result

    def validate(self, choice, run, public):
        projected = project_target_choice(choice,run['target_contract'])
        return super().validate(projected,run,public)
