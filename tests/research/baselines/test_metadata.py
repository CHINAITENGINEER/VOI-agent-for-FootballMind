from football_research.baselines.metadata import model_metadata
from football_research.baselines.api import PredictionService


def test_offline_models_do_not_require_odds():
    for name in ("frequency", "poisson", "dixon_coles", "elo", "logistic", "tree"):
        assert model_metadata(name)["requires_market"] is False
    assert model_metadata("market")["requires_market"] is True


def test_legacy_model_remains_loadable_and_describes_requirements():
    from pathlib import Path
    artifact = Path(__file__).resolve().parents[3] / "var/parallel/baselines/real-asof-final/models/1ac913520e-poisson.json"
    model = PredictionService.load(artifact)
    data = model.describe()
    assert data["training_matches"] == 66
    assert data["required_features"] == []
    assert data["training_cutoff"] == "2026-09-20T11:22:02+00:00"
