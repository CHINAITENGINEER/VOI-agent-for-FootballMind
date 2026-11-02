"""A frozen prior plus seven independently trained partial-information models.

Each nonempty subset has its own regularized multinomial model and preprocessing.
An unavailable group never reaches a fitted full-information model by imputation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from football_research.baselines.models import LeagueFrequency, LogisticBaseline, _training, _label
from .state import GROUP_FIELDS, GROUP_IDS, parse_time, sanitize_state

VERSION = "partial-subset-logistic-1.0"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class PartialPredictor:
    """Supervised fit is offline; predict_partial accepts visible state only."""

    version = VERSION

    def __init__(self) -> None:
        self.prior: LeagueFrequency | None = None
        self.models: dict[int, LogisticBaseline | None] = {}
        self.metadata: dict[str, Any] = {}
        self.prior_version = ""
        self.model_version = ""

    def fit(self, rows: Sequence[Mapping[str, Any]], as_of: Any, dataset_version: str,
            regularization: float = 10.0, min_train_matches: int = 15) -> "PartialPredictor":
        if not math.isfinite(regularization) or regularization <= 0 or min_train_matches < 1 or not dataset_version:
            raise ValueError("Invalid partial model configuration")
        selected = _training(rows, as_of)
        modes = {row["information_mode"] for row in selected}
        if len(modes) != 1 or not modes.issubset({"operational", "historical_reconstruction"}):
            raise ValueError("Train within exactly one information mode")
        for row in selected:
            cutoff = parse_time(row["prediction_cutoff"])
            if cutoff >= parse_time(row["kickoff"]):
                raise ValueError("Training features must be prematch")
            if row.get("features_available_at") is not None and parse_time(row["features_available_at"]) > cutoff:
                raise ValueError("Training features are unavailable at their cutoff")
            for fields in GROUP_FIELDS.values():
                for field in fields:
                    value = row.get(field)
                    if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)):
                        raise ValueError("Training fields must be finite numeric values or null")
        self.prior = LeagueFrequency(alpha=1.0).fit(selected, as_of)
        self.prior_version = "partial-frozen-prior-1.0:" + _hash(self.prior.to_state())
        training = [{"match_id": str(row["match_id"]), "prediction_cutoff": parse_time(row["prediction_cutoff"]).isoformat(),
                     "result_available_at": parse_time(row["result_available_at"]).isoformat()} for row in selected]
        self.metadata = {"dataset_version": dataset_version, "training_as_of": parse_time(as_of).isoformat(),
                         "information_mode": next(iter(modes)), "training_rows": len(selected),
                         "regularization": regularization, "min_train_matches": min_train_matches,
                         "group_fields": {k: list(v) for k, v in GROUP_FIELDS.items()}, "subsets": {}}
        self.models = {}
        classes = {_label(row) for row in selected}
        for mask in range(8):
            groups = [name for i, name in enumerate(GROUP_IDS) if mask & (1 << i)]
            fields = [field for name in groups for field in GROUP_FIELDS[name]]
            reason = "empty_mask_frozen_prior" if not mask else None
            if mask and len(selected) < min_train_matches:
                reason = "insufficient_training_matches"
            elif mask and classes != {0, 1, 2}:
                reason = "training_class_support"
            elif mask and any(all(row.get(field) is None for row in selected) for field in fields):
                reason = "training_feature_support"
            model = None if reason else LogisticBaseline(feature_names=fields, regularization=regularization).fit(selected, as_of)
            if mask:
                self.models[mask] = model
            self.metadata["subsets"][str(mask)] = {"groups": groups, "features": fields, "training": training,
                "model_kind": "league_frequency" if reason else "subset_multinomial_logistic", "fallback_reason": reason}
        self.model_version = VERSION + ":" + _hash(self._payload())
        return self

    def predict_partial(self, visible_state: Mapping[str, Any]) -> dict[str, Any]:
        if self.prior is None:
            raise ValueError("Fit partial model before prediction")
        state = sanitize_state(visible_state)
        if state["information_mode"] != self.metadata["information_mode"]:
            raise ValueError("Model and visible information modes differ")
        if parse_time(state["prediction_cutoff"]) < parse_time(self.metadata["training_as_of"]):
            raise ValueError("Prediction precedes model training cutoff")
        mask = state["mask_key"]
        model = self.models.get(mask)
        row = {"competition_id": state["base"]["competition_id"], "prediction_cutoff": state["prediction_cutoff"]}
        for group in state["groups"].values():
            row.update(group["values"])
        probs = self.prior.predict_proba(row) if model is None else model.predict_proba(row)
        return {"match_id": state["match_id"], "prediction_cutoff": state["prediction_cutoff"],
                "information_mode": state["information_mode"], "home_win_probability": probs[0],
                "draw_probability": probs[1], "away_win_probability": probs[2], "probabilities": list(probs),
                "mask": state["mask"], "mask_key": mask, "model_version": self.model_version,
                "prior_version": self.prior_version, "dataset_version": self.metadata["dataset_version"],
                "fallback_reason": self.metadata["subsets"][str(mask)]["fallback_reason"],
                "group_statuses": {k: v["status"] for k, v in state["groups"].items()},
                "visible_lineage": {k: {field: value for field, value in group.items() if field != "values"}
                                    for k, group in state["groups"].items() if group["status"] == "observed"}}

    def _payload(self) -> dict[str, Any]:
        return {"schema_version": 1, "implementation_version": VERSION, "metadata": self.metadata,
                "prior_version": self.prior_version, "prior": self.prior.to_state() if self.prior else None,
                "models": {str(k): v.to_state() if v else None for k, v in self.models.items()}}

    def to_state(self) -> dict[str, Any]:
        if self.prior is None:
            raise ValueError("Fit partial model before serialization")
        # Deep JSON copy prevents accidental mutation through exported structures.
        return json.loads(json.dumps({**self._payload(), "model_version": self.model_version}, allow_nan=False))

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "PartialPredictor":
        if state.get("schema_version") != 1 or state.get("implementation_version") != VERSION:
            raise ValueError("Unsupported partial model artifact")
        payload = {k: v for k, v in state.items() if k != "model_version"}
        if state.get("model_version") != VERSION + ":" + _hash(payload):
            raise ValueError("Partial model artifact content hash mismatch")
        if state["metadata"]["group_fields"] != {k: list(v) for k, v in GROUP_FIELDS.items()} or set(state["models"]) != {str(i) for i in range(1, 8)}:
            raise ValueError("Partial model subset schema mismatch")
        result = cls()
        result.prior = LeagueFrequency.from_state(state["prior"])
        result.prior_version = str(state["prior_version"])
        if result.prior_version != "partial-frozen-prior-1.0:" + _hash(result.prior.to_state()):
            raise ValueError("Frozen prior hash mismatch")
        result.models = {int(k): LogisticBaseline.from_state(v) if v else None for k, v in state["models"].items()}
        result.metadata = json.loads(json.dumps(state["metadata"], allow_nan=False))
        result.model_version = str(state["model_version"])
        return result

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_state(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PartialPredictor":
        return cls.from_state(json.loads(Path(path).read_text(encoding="utf-8")))
