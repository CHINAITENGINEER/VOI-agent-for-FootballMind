"""Same-f signed gains from original registered role-aware B executions."""
from .official_context_gain import _build_context_gain_from_runs

VERSION = 'official-role-context-gain-v1'
SOURCE_VERSION = 'raw-official-role-context-gain-inputs-v1'


def bind_role_visible(visible, binding, archive):
    from football_research.acquisition.fixture_roles import bind_fixture_roles
    from .fixture_role_identity import resolve_fixture_role_identity
    identity = resolve_fixture_role_identity(binding, archive_root=archive,
        as_of=visible['before']['decision_at'])
    return bind_fixture_roles(visible, identity)


def build_role_context_gain_from_runs(run_folders, result_archives, output, *, predictor_path, as_of=None):
    from football_research.acquisition.role_context_predictor import RoleContextPredictor
    from football_research.acquisition.role_context_query import query_input
    return _build_context_gain_from_runs(run_folders, result_archives, output,
        predictor_path=predictor_path, as_of=as_of, predictor_loader=RoleContextPredictor.load,
        query_builder=query_input, version=VERSION, source_version=SOURCE_VERSION,
        bind_visible=bind_role_visible)


__all__ = ['VERSION', 'SOURCE_VERSION', 'build_role_context_gain_from_runs']
