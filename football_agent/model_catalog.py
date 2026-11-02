"""Server-owned model allowlist; model IDs never double as filesystem paths.

The default entries pin the first-round, real-data baseline artifacts. An
operator may supply an explicit registry JSON containing ``models`` entries;
each artifact entry must include its integrity hash. HTTP clients supply only
an ID and never a registry or artifact path.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_INPUTS = ["match_id", "competition_id", "home_team_id", "away_team_id", "kickoff",
                  "fixture_available_at", "information_mode", "score_period", "input_manifest"]
_PINS = {
    "spain": ("97d56bbfb8", "spftll:competition:31", {
        "frequency": "90eccca92d3c2c5dab287e76ea1e43fc1ca09eeb68cb405e38fc0397cf1a021f",
        "poisson": "1e78dd4e6a6acd3e8f6c8a5bf473a3081e87312538711402bb1db11a56f4cc63",
        "dixon_coles": "aa660f89f4fb32b70cd48f19b5ac6e09111f73fe8826d3dff904b17a6524f215"}),
    "england": ("1ac913520e", "spftll:competition:36", {
        "frequency": "398b0197f779331736a1cf1a715b72a4d73072fa5b2ffd6094be6dcdf745dad3",
        "poisson": "f0cdf6cebc4569ac9466a266575e91fd2af461a16f38d6a208f84064a1892a64",
        "dixon_coles": "0404f5da9789a707c1cd2d76adcc88300b7571ecff6a05174dff16c407dd628c"}),
    "germany": ("2deee0ca27", "spftll:competition:8", {
        "frequency": "c527872b79dbdd37ebb50ccbe270e329773fc53d809db80797890411ff972748",
        "poisson": "a57d9099714d76fa4b6a82eb31a0d909fb9ea568b3c7c25fe0ff4512a5651c64",
        "dixon_coles": "dc4bc8f4b2cb2f8578df1f8a7cad65171f1275e4453846ffcc3fe969d418e8d0"}),
}


def required_inputs(model_name: str, model=None, service=None) -> dict:
    """Prediction requirements, deliberately separate from training fields."""
    from football_research.baselines.metadata import model_metadata
    spec = model_metadata(model_name, service=service)
    feature_fields = spec["required_features"]
    if service is None and model is not None and model_name in {"logistic", "tree"}:
        feature_fields = list(model.feature_names)
    return {"fixture": list(FIXTURE_INPUTS), "features": feature_fields,
            "feature_snapshot_required": spec["requires_feature_snapshot"],
            "odds_required": spec["requires_market"],
            "source": "football_research.baselines.metadata.model_metadata",
            "trained_model_required": model_name != "market"}


class ModelCatalog:
    def __init__(self, registry_path=None):
        self.registry_path = str(Path(registry_path).resolve()) if registry_path else None
        entries = [{"model_id": "market", "model_name": "market", "model_artifact": None,
                    "information_mode": "operational", "competition_id": None}]
        if registry_path is None:
            for league, (prefix, competition, pins) in _PINS.items():
                for model, pin in pins.items():
                    entries.append({"model_id": model.replace("_", "-") + "-" + league,
                                    "model_name": model, "competition_id": competition,
                                    "information_mode": "operational", "artifact_sha256": pin,
                                    "model_artifact": str(ROOT / "var/parallel/baselines/real-asof-final/models" /
                                                          f"{prefix}-{model}.json")})
        else:
            payload = json.loads(Path(self.registry_path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
                raise ValueError("Model registry requires a models array")
            entries = payload["models"]
        self._entries = {}
        for raw in entries:
            entry = deepcopy(raw)
            mid = entry.get("model_id")
            if not isinstance(mid, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,79}", mid):
                raise ValueError("Model IDs must be simple allowlist identifiers")
            if mid in self._entries:
                raise ValueError("Duplicate model registry ID")
            artifact = entry.get("model_artifact")
            if artifact:
                if not re.fullmatch(r"[a-f0-9]{64}", str(entry.get("artifact_sha256", ""))):
                    raise ValueError("Registered artifacts require a pinned SHA256")
                path = Path(artifact)
                if not path.is_absolute():
                    path = (Path(self.registry_path).parent if self.registry_path else ROOT) / path
                entry["model_artifact"] = str(path.resolve())
            elif entry.get("model_name") != "market":
                raise ValueError("Only the stateless market model may omit an artifact")
            self._entries[mid] = entry

    def resolve(self, model_id: str) -> dict:
        if not isinstance(model_id, str) or model_id not in self._entries:
            raise ValueError("Unknown model ID; choose a server-registered model")
        from football_research.baselines.api import PredictionService, model_registry
        entry = deepcopy(self._entries[model_id])
        model_name = entry["model_name"]
        if model_name not in model_registry():
            raise ValueError("Registered model is unsupported by PredictionService")
        if entry.get("model_artifact"):
            service = PredictionService.load(entry["model_artifact"])
            artifact = service.to_artifact()
            if artifact["artifact_sha256"] != entry["artifact_sha256"]:
                raise ValueError("Registered model hash differs from the server allowlist")
            if service.model_name != model_name or service.information_mode != entry["information_mode"]:
                raise ValueError("Registered model name/information mode differs from its artifact")
            entry.update(model_version=artifact["model_state"]["model_version"],
                         trained_at=service.trained_at.isoformat(), dataset_version=service.dataset_version,
                         training_status=service.training_manifest.get("mapping_status", "fitted_artifact"), training_matches=service.training_manifest["training_matches"],
                         required_inputs=required_inputs(model_name, service.model, service=service),
                         prediction_service_metadata=service.describe())
        else:
            model_type = model_registry()[model_name]
            code_path = ROOT / "football_research/baselines/models.py"
            reference = {"model_name": model_name, "model_version": model_type.version,
                         "parameters": model_type().to_state()["parameters"],
                         "code_sha256": sha256(code_path.read_bytes()).hexdigest()}
            entry.update(model_version=model_type.version, trained_at=None, dataset_version=None,
                         artifact_sha256=sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
                         training_status="stateless_at_prediction_cutoff", training_matches=0,
                         required_inputs=required_inputs(model_name),
                         supported_information_modes=["operational", "historical_reconstruction"])
        entry.update(status="OK", error=None)
        return entry

    def list_models(self) -> list[dict]:
        result = []
        for model_id, raw in self._entries.items():
            try:
                result.append(self.resolve(model_id))
            except (ImportError, OSError, ValueError, KeyError) as exc:
                result.append({**deepcopy(raw), "status": "MODEL_UNAVAILABLE", "error": str(exc),
                               "model_version": None, "trained_at": None,
                               "required_inputs": required_inputs(raw["model_name"])})
        return result
