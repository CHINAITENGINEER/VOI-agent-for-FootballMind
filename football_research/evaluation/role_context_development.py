"""Explicit role-f variant of the same whole-fixture retrospective protocol."""
from football_research.acquisition.fixture_roles import role_state
from football_research.acquisition.role_context_predictor import RoleContextPredictor, training_readiness
from .optional_context_development import _evaluate_context_development

VERSION = 'role-context-retrospective-development-v1'


def evaluate_role_context_development(records, *, training_cutoff, as_of, evidence_origin, dataset_version):
    result = _evaluate_context_development(records, training_cutoff=training_cutoff, as_of=as_of,
        evidence_origin=evidence_origin, dataset_version=dataset_version,
        model_factory=RoleContextPredictor, readiness_function=training_readiness, state_validator=role_state,
        representation_limitation='Verified aliases bind observed text mentions to home/away roles; lexical association does not establish predicate/event argument understanding.')
    result['schema'] = VERSION
    result['representation'] = 'verified_role_by_text_channels_same_fixed_optional_offset_predictor_family'
    if result['summary'] is not None:
        for position in ('initial', 'terminal'):
            result['summary'][position]['role_extrapolation_physical_matches'] = sum(
                bool(item[position]['forecast']['role_support']['unseen_training_roles'])
                for item in result['per_match'])
    return result


__all__ = ['evaluate_role_context_development', 'VERSION']
