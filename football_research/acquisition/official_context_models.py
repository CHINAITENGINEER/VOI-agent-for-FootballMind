"""Explicit artifact dispatch; original optional-context models stay unchanged."""
import json
from pathlib import Path

from .optional_prior_predictor import OptionalPriorContextPredictor

ROLE_PREDICTOR = 'role-optional-prior-visible-context-v1'


def is_role_predictor(model):
    if getattr(model, 'version', None) != ROLE_PREDICTOR:
        return False
    from .role_context_predictor import RoleContextPredictor
    return isinstance(model, RoleContextPredictor)


def load_official_predictor(path):
    schema = json.loads(Path(path).read_text(encoding='utf-8')).get('schema')
    if schema == ROLE_PREDICTOR:
        from .role_context_predictor import RoleContextPredictor
        return RoleContextPredictor.load(path)
    return OptionalPriorContextPredictor.load(path)


def load_official_value(path, *, predictor_path):
    predictor = load_official_predictor(predictor_path)
    if is_role_predictor(predictor):
        from .role_context_value import RoleContextValueModel
        return RoleContextValueModel.load(path, predictor_path=predictor_path)
    from .optional_context_value import OptionalContextValueModel
    return OptionalContextValueModel.load(path, predictor_path=predictor_path)


__all__ = ['ROLE_PREDICTOR', 'is_role_predictor', 'load_official_predictor', 'load_official_value']
