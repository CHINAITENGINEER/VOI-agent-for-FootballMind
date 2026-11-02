"""Small Agent-facing prediction service; all timestamps are evidence, not row dates.

Prediction extends the existing platform Contract until the contracts owner
publishes the probability contract requested in docs/parallel/requests/baselines.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import AwareDatetime, Field, model_validator
from football_research.data_adapters._contracts import Contract, SCHEMA_VERSION

API_VERSION = "baselines-api-v1"
PROBABILITY_KEYS = ("home_win_probability", "draw_probability", "away_win_probability")


def utc(value: datetime | str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("An explicit timezone is required")
    return result.astimezone(timezone.utc)


def canonical_json(value: Any) -> str:
    def encode(item):
        if isinstance(item, datetime):
            return utc(item).isoformat()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=encode, allow_nan=False)


def content_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def check_feature_availability(features: Mapping, cutoff, *, require=False):
    fields = ("features_available_at", "elo_available_at")
    known = [features[k] for k in fields if features.get(k) is not None]
    if require and not known:
        raise ValueError("Numeric features require explicit availability evidence")
    if any(utc(value) > utc(cutoff) for value in known):
        raise ValueError("Features are unavailable at prediction cutoff")


class Prediction(Contract):
    match_id: str = Field(min_length=1)
    prediction_cutoff: AwareDatetime
    home_win_probability: float = Field(ge=0, le=1)
    draw_probability: float = Field(ge=0, le=1)
    away_win_probability: float = Field(ge=0, le=1)
    model_version: str
    dataset_version: str
    input_manifest: dict[str, Any]
    information_mode: Literal["operational", "historical_reconstruction"]

    @model_validator(mode="after")
    def probabilities_sum_to_one(self):
        if not math.isclose(sum(getattr(self, k) for k in PROBABILITY_KEYS), 1., abs_tol=1e-10):
            raise ValueError("Probabilities must sum to one")
        return self


def model_registry():
    from .models import LeagueFrequency, PoissonModel, DixonColesModel, MarketBaseline, EloMapping, LogisticBaseline, TreeBaseline
    return {"frequency": LeagueFrequency, "poisson": PoissonModel,
            "dixon_coles": DixonColesModel, "market": MarketBaseline, "elo": EloMapping,
            "logistic": LogisticBaseline, "tree": TreeBaseline}


def training_rows(rows: Sequence[Mapping], *, as_of, information_mode: str,
                  excluded_match_ids=()) -> list[dict]:
    """Unique completed regulation outcomes visible strictly before model freeze.

    A forecast at freeze time must not use a result arriving at that same instant.
    Unknown visibility is excluded. Mixed modes are rejected, not silently merged.
    """
    cutoff = utc(as_of)
    selected = {}
    excluded = set(excluded_match_ids)
    for row in rows:
        if row.get("information_mode") != information_mode:
            raise ValueError("Mixed or unspecified information modes in dataset")
        if row["match_id"] in excluded or row.get("score_period") != "regulation":
            continue
        if row.get("home_goals") is None or row.get("away_goals") is None or row.get("result_available_at") is None:
            continue
        for key in ("home_goals", "away_goals"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("Scores must be nonnegative integers")
        available = utc(row["result_available_at"])
        kickoff = utc(row["kickoff"])
        if available < kickoff:
            raise ValueError("Result availability precedes kickoff")
        if available >= cutoff or kickoff >= cutoff:
            continue
        item = dict(row, kickoff=kickoff, result_available_at=available)
        previous = selected.get(row["match_id"])
        if previous is not None:
            fields = ("home_goals", "away_goals", "home_team_id", "away_team_id", "competition_id", "kickoff")
            if any(previous[k] != item[k] for k in fields):
                raise ValueError("Conflicting match versions require upstream resolution")
        else:
            selected[row["match_id"]] = item
    return sorted(selected.values(), key=lambda r: (r["kickoff"], r["match_id"]))


class PredictionService:
    """Train once on a frozen window, predict many fixtures; safe JSON persistence."""

    def __init__(self, model_name: str, model, *, trained_at, dataset_version: str,
                 information_mode: str, training_manifest: dict):
        self.model_name, self.model = model_name, model
        self.trained_at = utc(trained_at)
        self.dataset_version = dataset_version
        self.information_mode = information_mode
        self.training_manifest = training_manifest

    @classmethod
    def train(cls, model_name: str, rows: Sequence[Mapping], *, as_of, dataset_version: str,
              information_mode="operational", parameters=None, excluded_match_ids=()):
        registry = model_registry()
        if model_name not in registry:
            raise ValueError(f"Unknown model {model_name!r}")
        if model_name in {"logistic", "tree"}:
            from football_research.data_adapters.batch_features import _safe_feature
            names = (parameters or {}).get("feature_names", ("home_elo", "away_elo"))
            if any(not _safe_feature(name) for name in names):
                raise ValueError("Disallowed target/final/forecast numeric features")
        eligible = training_rows(rows, as_of=as_of, information_mode=information_mode,
                                 excluded_match_ids=excluded_match_ids)
        if model_name == "market":
            eligible = []  # Market probabilities consume only the current quote.
        if model_name in {"elo", "logistic", "tree"}:
            versions = {row.get("feature_version") for row in eligible if row.get("feature_version")}
            if len(versions) > 1 or (versions and any(not row.get("feature_version") for row in eligible)):
                raise ValueError("Mixed feature versions in training data")
            for row in eligible:
                if row.get("prediction_cutoff") is None or utc(row["prediction_cutoff"]) >= utc(row["kickoff"]):
                    raise ValueError("Supervised feature rows require their own prematch prediction_cutoff")
                if not row.get("feature_input_manifest"):
                    raise ValueError("Supervised feature rows require feature_input_manifest")
                check_feature_availability(row, row["prediction_cutoff"], require=True)
        if model_name not in {"market", "elo"} and not eligible:
            raise ValueError("No eligible historical training outcomes")
        model = registry[model_name](**(parameters or {}))
        model.fit(eligible, as_of=utc(as_of))
        manifest = {"api_version": API_VERSION, "platform_schema_version": SCHEMA_VERSION,
                    "train_cutoff": utc(as_of).isoformat(), "training_matches": len(eligible),
                    "training_match_ids": [r["match_id"] for r in eligible],
                    "training_input_hash": content_hash(eligible),
                    "training_inputs": sorted({str(ref) for r in eligible for ref in r.get("input_manifest", [])}),
                    "model_parameters": model.to_state()["parameters"],
                    "code_sha256": sha256(Path(__file__).with_name("models.py").read_bytes()).hexdigest(),
                    "target": "regulation_plus_stoppage_excludes_extra_time_penalties"}
        if model_name in {"elo", "logistic", "tree"}:
            manifest["feature_version"] = next(iter(versions), None)
            manifest["training_feature_inputs"] = [{"match_id": row["match_id"],
                "prediction_cutoff": utc(row["prediction_cutoff"]).isoformat(),
                "feature_input_manifest_sha256": content_hash(row["feature_input_manifest"]),
                "feature_snapshot_sha256": row["feature_input_manifest"].get("feature_snapshot_sha256")
                    if isinstance(row["feature_input_manifest"], Mapping) else None} for row in eligible]
        return cls(model_name, model, trained_at=as_of, dataset_version=dataset_version,
                   information_mode=information_mode, training_manifest=manifest)

    def predict(self, fixture: Mapping, *, prediction_cutoff, features: Mapping | None = None) -> Prediction:
        cutoff = utc(prediction_cutoff)
        if fixture.get("information_mode") != self.information_mode:
            raise ValueError("Prediction and training information modes differ")
        if fixture.get("status") in {"postponed", "cancelled", "suspended"}:
            raise ValueError("Inactive fixture requires a resolved schedule version")
        if fixture.get("score_period") != "regulation":
            raise ValueError("Target regulation-time semantics must be established")
        if self.trained_at > cutoff:
            raise ValueError("Model was trained after prediction cutoff")
        if cutoff >= utc(fixture["kickoff"]):
            raise ValueError("Prediction must precede kickoff")
        if fixture.get("fixture_available_at") is None or utc(fixture["fixture_available_at"]) > cutoff:
            raise ValueError("Fixture was not known at prediction cutoff")
        if fixture["match_id"] in self.training_manifest["training_match_ids"]:
            raise ValueError("Target match occurs in model training inputs")
        row = {k: fixture[k] for k in ("match_id", "competition_id", "home_team_id", "away_team_id", "kickoff")}
        row.update(prediction_cutoff=cutoff)
        feature_manifest = None
        if features is None and self.model_name in {"elo", "logistic", "tree"}:
            raise ValueError("A timestamped feature snapshot is required")
        if features is not None:
            expected_feature_version = self.training_manifest.get("feature_version")
            if expected_feature_version and features.get("feature_version") != expected_feature_version:
                raise ValueError("Feature version differs from model training version")
            if str(features.get("match_id")) != str(fixture["match_id"]):
                raise ValueError("Features belong to another match")
            if features.get("information_mode") != self.information_mode:
                raise ValueError("Feature information mode differs")
            if utc(features["prediction_cutoff"]) != cutoff:
                raise ValueError("Features must share the exact prediction cutoff")
            if not features.get("input_manifest"):
                raise ValueError("Features require input lineage")
            check_feature_availability(features, cutoff, require=self.model_name in {"elo", "logistic", "tree"})
            allowed = ("home_elo", "away_elo", "elo_available_at", "features_available_at", "odds_home", "odds_draw", "odds_away", "odds_available_at",
                       "odds_source_timestamp", "odds_period", "odds_market_type", "odds_snapshot_id", "odds_line", "odds_selection_timestamps")
            if self.model_name in {"logistic", "tree"}:
                allowed = (*allowed, *self.model.feature_names)
            row.update({k: features[k] for k in allowed if k in features})
            feature_manifest = features["input_manifest"]
        probabilities = self.model.predict_proba(row)
        state = self.model.to_state()
        training_references = {k: v for k, v in self.training_manifest.items() if k != "training_feature_inputs"}
        if "training_feature_inputs" in self.training_manifest:
            training_references["training_feature_inputs_sha256"] = content_hash(self.training_manifest["training_feature_inputs"])
        manifest = {**training_references, "model_state_sha256": content_hash(state),
                    "target_input": list(fixture.get("input_manifest", [])),
                    "target_fixture_sha256": content_hash({k: fixture.get(k) for k in
                        ("match_id", "competition_id", "home_team_id", "away_team_id", "kickoff", "fixture_available_at")}),
                    "feature_input": feature_manifest}
        version = str(state.get("model_version", state.get("version", self.model_name + "-v1")))
        return Prediction(match_id=fixture["match_id"], prediction_cutoff=cutoff,
                          **dict(zip(PROBABILITY_KEYS, probabilities)), model_version=version,
                          dataset_version=self.dataset_version, input_manifest=manifest,
                          information_mode=self.information_mode)

    @classmethod
    def fixed_elo_mapping(cls, *, as_of, dataset_version: str, information_mode="operational", parameters=None):
        """Explicit unfitted Davidson map; upstream features own the Elo ratings."""
        from .models import EloMapping
        model = EloMapping(**(parameters or {}))
        manifest = {"api_version": API_VERSION, "platform_schema_version": SCHEMA_VERSION,
                    "train_cutoff": utc(as_of).isoformat(), "training_matches": 0, "training_match_ids": [],
                    "training_inputs": [], "training_input_hash": content_hash([]),
                    "mapping_status": "fixed_parameters_not_fitted", "rating_owner": "features",
                    "model_parameters": model.to_state()["parameters"],
                    "code_sha256": sha256(Path(__file__).with_name("models.py").read_bytes()).hexdigest(),
                    "target": "regulation_plus_stoppage_excludes_extra_time_penalties"}
        return cls("elo", model, trained_at=as_of, dataset_version=dataset_version,
                   information_mode=information_mode, training_manifest=manifest)

    def to_artifact(self) -> dict:
        result = {"api_version": API_VERSION, "model_name": self.model_name, "trained_at": self.trained_at.isoformat(),
                  "dataset_version": self.dataset_version, "information_mode": self.information_mode,
                  "training_manifest": self.training_manifest, "model_state": self.model.to_state()}
        return {**result, "artifact_sha256": content_hash(result)}

    def describe(self) -> dict:
        """Model-specific input requirements; compatible with existing artifacts."""
        from .metadata import model_metadata
        return model_metadata(service=self)

    def save(self, path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(canonical_json(self.to_artifact()) + "\n", encoding="utf-8")
        return dest

    @classmethod
    def load(cls, path):
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = artifact.pop("artifact_sha256")
        if content_hash(artifact) != expected or artifact["api_version"] != API_VERSION:
            raise ValueError("Artifact integrity/version check failed")
        model = model_registry()[artifact["model_name"]].from_state(artifact["model_state"])
        return cls(artifact["model_name"], model, trained_at=artifact["trained_at"],
                   dataset_version=artifact["dataset_version"], information_mode=artifact["information_mode"],
                   training_manifest=artifact["training_manifest"])
