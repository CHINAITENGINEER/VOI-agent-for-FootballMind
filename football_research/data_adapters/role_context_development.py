"""Raw temporal A evaluation for the explicit fixture-role representation."""
from .optional_context_development import _evaluate_context_from_runs


def evaluate_role_context_from_runs(run_folders, previous_plan_paths, result_archive, output,
                                   *, training_cutoff, as_of=None):
    return _evaluate_context_from_runs(run_folders, previous_plan_paths, result_archive, output,
        training_cutoff=training_cutoff, as_of=as_of, role_context=True)


__all__ = ['evaluate_role_context_from_runs']
