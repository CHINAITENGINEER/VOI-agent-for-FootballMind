'''Single-call candidate proposer; ranking/selection remains a separate policy.'''
from copy import deepcopy

from .experience import (MatchAnalysisProtocol, MatchLocalAnalysisProtocol,
                         MatchReturnAnalysisProtocol, MatchLocalReturnAnalysisProtocol)
from .experience_provider import ExperienceProvider
from .acquisition_boundary import SHARED_FIELDS, OPTIONAL_SHARED_FIELDS
from . import match_return_assessment as return_review
from . import runtime_source_claims as source_claims


def proposal_schema(protocol=MatchAnalysisProtocol):
    original=protocol.schema['properties']
    fields = SHARED_FIELDS + tuple(key for key in OPTIONAL_SHARED_FIELDS if key in original)
    shared={key:deepcopy(original[key]) for key in fields}
    action={key:deepcopy(value) for key,value in original.items() if key not in fields and key!='alternatives'}
    shared.update(candidates={'type':'array','minItems':1,'maxItems':6,'items':{
        'type':'object','properties':action,'required':list(action),'additionalProperties':False}},
        recommended_index={'type':'integer','minimum':0,'maximum':5})
    return {'type':'object','properties':shared,'required':list(shared),'additionalProperties':False}


class MatchProposalProtocol:
    version='match-candidate-proposal-v1'
    schema=proposal_schema()
    prompt=MatchAnalysisProtocol.prompt+'''
CANDIDATE PROPOSER CONTRACT (output shape overrides the single-decision format):
You interpret existing evidence and propose executable next actions. A separate
policy will select a candidate or stop. Return shared task_id, previous_return,
assessment, analysis_plan and match_identity ONCE at the top level. Return 1-6
distinct candidates with the remaining action fields, without alternatives.
When meaningful alternative acquisition actions exist, propose more than one;
do not add irrelevant actions to fill a quota. Each acquisition must address an
open question in analysis_plan, use a known URL or a new search query, and state
what return would matter. Do not treat a recommendation as executed.
recommended_index is a zero-based qualitative recommendation only, within the
returned candidate list. It supplies the old baseline, not numerical VoI.
All candidate assessments share the same actual evidence; never imagine future
source returns, outcomes, gains or costs. The executor validates every candidate.
The final choice and any algorithm diagnostics are separate from this response.
'''


class MatchLocalProposalProtocol:
    version='match-candidate-proposal-local-v1'
    schema=proposal_schema(MatchLocalAnalysisProtocol)
    prompt=MatchLocalAnalysisProtocol.prompt + MatchProposalProtocol.prompt[len(MatchAnalysisProtocol.prompt):]


class MatchReturnProposalProtocol:
    version='match-candidate-proposal-return-v1'
    schema=proposal_schema(MatchReturnAnalysisProtocol)
    prompt=MatchReturnAnalysisProtocol.prompt + MatchProposalProtocol.prompt[len(MatchAnalysisProtocol.prompt):] + ' Return return_assessment ONCE at top level; candidates share this actual-return assessment.'


class MatchLocalReturnProposalProtocol:
    version='match-candidate-proposal-local-return-v1'
    schema=proposal_schema(MatchLocalReturnAnalysisProtocol)
    prompt=MatchLocalReturnAnalysisProtocol.prompt + MatchProposalProtocol.prompt[len(MatchAnalysisProtocol.prompt):] + ' Return return_assessment ONCE at top level; candidates share this actual-return assessment.'


class MatchCandidateProvider:
    version=MatchProposalProtocol.version

    def __init__(self,folder,effort='low',*,provider=None):
        self.provider=provider
        self.last_provider = provider
        self.folder,self.effort=folder,effort

    def propose(self,public):
        if public.get('task_type')!='match_analysis':
            raise ValueError('MATCH_PROPOSER_REQUIRES_MATCH_TASK')
        protocol=MatchLocalProposalProtocol if public.get('local_query_capability') else MatchProposalProtocol
        if return_review.CAPABILITY_FIELD in public:
            return_review.ensure_capability(public)
            protocol=MatchLocalReturnProposalProtocol if public.get('local_query_capability') else MatchReturnProposalProtocol
        if source_claims.CAPABILITY_FIELD in public:
            source_claims.ensure_capability(public)
            protocol=source_claims.extend_protocol(protocol,
                source_support='source_support_capability' in public)
        from . import runtime_local_acquisition as local_acquisition
        if local_acquisition.enabled(public):
            protocol=local_acquisition.extend_protocol(protocol)
        provider=self.provider if self.provider is not None else ExperienceProvider(
            self.folder,protocol=protocol,effort=self.effort,timeout_seconds=60)
        self.last_provider = provider
        result=provider.decide(public)
        response=result['response']
        index=response['recommended_index']
        if type(index) is not int or not 0<=index<len(response['candidates']):
            raise ValueError('PROPOSER_RECOMMENDATION_OUT_OF_RANGE')
        shared={key:deepcopy(response[key]) for key in SHARED_FIELDS}
        shared.update({key:deepcopy(response[key]) for key in OPTIONAL_SHARED_FIELDS if key in response})
        candidates=[{**shared,**deepcopy(action),'alternatives':[]} for action in response['candidates']]
        return {'candidates':candidates,'recommended_index':index,
                'proposal_protocol':protocol.version,'raw_provider_result':result}
