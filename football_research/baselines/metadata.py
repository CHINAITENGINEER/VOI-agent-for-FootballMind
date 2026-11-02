"""Machine-readable model requirements for Agent and batch consumers."""
from __future__ import annotations
from copy import deepcopy


_SPECS = {
    "frequency": {"required_features": [], "requires_market": False, "fit_requires": ["competition_id", "home_goals", "away_goals"]},
    "poisson": {"required_features": [], "requires_market": False, "fit_requires": ["competition_id", "home_team_id", "away_team_id", "home_goals", "away_goals"]},
    "dixon_coles": {"required_features": [], "requires_market": False, "fit_requires": ["competition_id", "home_team_id", "away_team_id", "home_goals", "away_goals"]},
    "market": {"required_features": ["odds_home", "odds_draw", "odds_away", "odds_available_at"], "requires_market": True, "fit_requires": []},
    "elo": {"required_features": ["home_elo", "away_elo"], "requires_market": False, "fit_requires": ["home_elo", "away_elo", "home_goals", "away_goals"]},
    "logistic": {"required_features": ["home_elo", "away_elo"], "requires_market": False, "fit_requires": ["home_goals", "away_goals"], "configurable_numeric_features": True},
    "tree": {"required_features": ["home_elo", "away_elo"], "requires_market": False, "fit_requires": ["home_goals", "away_goals"], "configurable_numeric_features": True},
}


def model_metadata(model_name=None, *, service=None):
    """Describe names or a loaded artifact without requiring quotes to exist."""
    from .api import model_registry
    registry = model_registry()
    if model_name is None and service is None:
        return {name: model_metadata(name) for name in registry}
    name = service.model_name if service is not None else model_name
    if name not in registry:
        raise ValueError(f"Unknown model {name!r}")
    spec = deepcopy(_SPECS[name])
    if service is not None and name in {"logistic", "tree"}:
        spec["required_features"] = list(service.model.feature_names)
    spec.update(model_name=name, model_version=registry[name].version,
                requires_feature_snapshot=name in {"elo", "logistic", "tree", "market"},
                supported_information_modes=["operational", "historical_reconstruction"],
                target="regulation_plus_stoppage_excludes_extra_time_penalties",
                probability_order=["home", "draw", "away"],
                id_namespace_policy="exact_training_dataset_ids_or_explicit_evidenced_mapping",
                missing_policy="reject_missing_snapshot; numeric_missing_values_use_training_imputer_only_for_logistic_tree",
                training_cutoff=service.trained_at.isoformat() if service else None,
                dataset_version=service.dataset_version if service else None,
                information_mode=service.information_mode if service else None)
    if service is not None:
        spec["training_matches"] = service.training_manifest["training_matches"]
        ids = service.training_manifest.get("training_match_ids", [])
        spec["id_namespaces"] = service.training_manifest.get("id_namespaces", sorted({
            value.rsplit(":", 1)[0] if ":" in value else "canonical_or_external_id" for value in ids}) or ["caller_fixture_namespace"])
        spec["parameters"] = service.model.to_state()["parameters"]
    return spec
