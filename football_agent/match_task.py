'''Match-first task definition: users identify the event; the agent owns inquiry design.'''
from copy import deepcopy

KIND = 'match_analysis'
GOAL = ('基于可追溯证据分析这场比赛的双方相对实力、可能赛况与关键不确定性；'
        '自主发现哪些未知可能改变分析，按预期决策价值与获取成本选择下一条信息，'
        '在继续获取不值得或预算不足时停止，说明已知、条件性判断与未解决事项。')

MATCH_PROMPT = '''
MATCH-FIRST TASK CONTRACT (overrides question-inquiry instructions):
The user supplies only a match, not a research question or a list of features.
The system supplies analysis_objective. Your job is to analyze that match and
autonomously discover which unknowns could change that analysis. Do not ask the
user what to research, which dimensions to prioritize, what information matters,
which sources to use, or how to interpret the match. Do not reduce the task to
answering one user-formulated question. Do not use a mandatory checklist of
injuries/weather/rankings: propose context-dependent questions yourself and
revise them after actual observations. Identifying the correct fixture is part
of your work. Search public sources to resolve the fixture before asking the
user for a missing date, competition or team identity. User clarification is
only for genuine fixture ambiguity or missing identity that remains after a
source attempt. Never fabricate a match date or treat a historical final score
as a pre-match prediction. For past fixtures, distinguish retrospective analysis
from what was knowable before kickoff; missing publication evidence stays unknown.
Return match_identity, an analysis_plan with at most 6 self-generated questions,
and active_question_id linking each source action to one open/uncertain question.
Each question has a concrete decision_relevance: which part of the match analysis
could change if answered. Plans may change; previous_analysis_plan is evidence of
your prior plan, not an instruction to keep investigating it. Summarize overall
match analysis in assessment.answer, preserving unresolved uncertainty. Merely
identifying the fixture is not a completed match analysis. No invented calibrated
VoI, win probabilities, utility weights, or numerical gains. The active selector
is declared in public.selection_policy: the baseline uses your qualitative
recommendation; a separate value selector can choose another proposed action
or defer. When fixture_identity is present, its archive-bound resolution is
separate from your match_identity label. A bounded identity prerequisite may
choose a proposed catalogue source before empirical value selection is possible;
this is explicitly unmeasured, not a calibrated positive VoI. Statements about
rotation, qualification and tactical choices are hypotheses until supported.
Set clarification_field to none unless action=ask_user. For ask_user use only
date, competition or teams, and match_identity.status must not be identified.
Set active_question_id to null for stop/defer/ask_user. Stop at a justified partial
analysis when additional information is not worth acquiring; do not make the
user decide your research scope or stopping threshold.
source_catalog lists optional pre-registered read_url capabilities. You may read
an exact catalog URL without asking the user for a source or searching first.
Choose it only if its stated coverage can help an open question; no source is
mandatory. Detail/odds URLs must be exact links returned after an observation,
never invented from team names or guessed match IDs. A catalog is capability
metadata, not evidence or proof of availability. platform_structured excerpts
are parsed source assertions with explicit missingness and incomplete coverage.
No records means this source returned no usable records, not that a match does
not exist. Current observations, historical records and source publication dates
are distinct. Bookmaker/market/settlement fields must retain their own meaning.
'''

def task_config(values):
    value = values.get('match')
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 600:
        raise ValueError('请输入一场比赛（3—600 字符，可包含双方、赛事和日期）')
    if 'question' in values or 'seed_urls' in values:
        raise ValueError('比赛分析只接收比赛信息；分析问题和来源由系统自行选择')
    value = value.strip()
    return {'task_type': KIND, 'match_input': value, 'analysis_objective': GOAL,
            'question': '自主分析：' + value, 'seed_urls': []}

def match_schema(base):
    schema = deepcopy(base)
    string = {'type': 'string', 'minLength': 1}
    def obj(properties):
        return {'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}
    properties = {
        'match_identity': obj({'status': {'type': 'string', 'enum': ['unresolved', 'ambiguous', 'identified']},
                               'label': string}),
        'analysis_plan': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': obj({
            'id': {'type': 'string', 'pattern': '^[a-zA-Z0-9_-]{1,32}$'},
            'question': string, 'decision_relevance': string,
            'status': {'type': 'string', 'enum': ['open', 'uncertain', 'supported', 'contradicted']}})},
        'active_question_id': {'type': ['string', 'null']},
        'clarification_field': {'type': 'string', 'enum': ['none', 'date', 'competition', 'teams']},
    }
    schema['properties'].update(properties)
    schema['required'].extend(properties)
    return schema

def validate_match_decision(response, public):
    plan = response['analysis_plan']
    questions = {q['id']: q for q in plan}
    if len(questions) != len(plan):
        raise ValueError('DUPLICATE_ANALYSIS_QUESTION')
    if response['action'] in {'search_web', 'read_url', 'query_acquired'}:
        selected = questions.get(response['active_question_id'])
        if not selected or selected['status'] not in {'open', 'uncertain'}:
            raise ValueError('SOURCE_ACTION_REQUIRES_OPEN_ANALYSIS_QUESTION')
    elif response['active_question_id'] is not None:
        raise ValueError('NO_SOURCE_ACTION_ACTIVE_QUESTION')
    if response['action'] == 'ask_user':
        if (response['clarification_field'] == 'none'
                or response['match_identity']['status'] == 'identified'):
            raise ValueError('ONLY_MATCH_IDENTITY_CLARIFICATION_ALLOWED')
        if not public.get('documents'):
            raise ValueError('TRY_IDENTIFYING_MATCH_BEFORE_ASKING_USER')
        question = response['question_to_user'] or ''
        scope_requests = ['关注哪', '重点分析', '分析哪些', '优先查', '查哪些', '想了解什么',
                          '希望了解', '提供来源', '提供链接', 'which aspect', 'which sources',
                          'what to research', 'what information', 'what would you like']
        if any(term in question.lower() for term in scope_requests):
            raise ValueError('DO_NOT_DELEGATE_ANALYSIS_SCOPE_TO_USER')
    elif response['clarification_field'] != 'none':
        raise ValueError('NO_CLARIFICATION_FIELD_WITHOUT_QUESTION')
