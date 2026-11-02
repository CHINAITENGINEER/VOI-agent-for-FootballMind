"""Auditable regulation-time H/D/A models, independent of storage or collectors.

Inputs are adapter-owned mappings. Fitted labels must already refer to regulation
time, including stoppage time. With ``as_of`` set, result availability is checked
again here. Probabilities always follow the order home, draw, away. JSON state
contains parameters and fitted values; loading never executes serialized code.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, softmax
from scipy.stats import poisson

Row = Mapping[str, Any]
Probabilities = tuple[float, float, float]
STATE_VERSION = 1


def _time(value: Any) -> datetime:
    stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Timestamps require an explicit timezone")
    return stamp.astimezone(timezone.utc)


def _probabilities(values: Sequence[float]) -> Probabilities:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr < 0) or arr.sum() <= 0:
        raise ValueError("Expected three finite nonnegative probabilities with positive mass")
    arr /= arr.sum()
    return tuple(float(v) for v in arr)


def _goals(row: Row) -> tuple[int, int]:
    values = (row.get("home_goals"), row.get("away_goals"))
    if any(isinstance(v, bool) or v is None or not math.isfinite(float(v)) or float(v) < 0 or float(v) != int(v) for v in values):
        raise ValueError("Training requires known nonnegative integer regulation goals")
    if row.get("score_period", "regulation") != "regulation":
        raise ValueError("Only regulation-time labels are supported")
    return int(values[0]), int(values[1])


def _label(row: Row) -> int:
    home, away = _goals(row)
    return 0 if home > away else 1 if home == away else 2


def _training(rows: Sequence[Row], as_of: Any = None) -> list[Row]:
    selected = list(rows)
    if not selected:
        raise ValueError("At least one eligible training match is required")
    cutoff = _time(as_of) if as_of is not None else None
    seen: set[str] = set()
    for row in selected:
        _goals(row)
        if row.get("match_id") is not None:
            mid = str(row["match_id"])
            if mid in seen:
                raise ValueError("Duplicate training match; group snapshots before fitting")
            seen.add(mid)
        if cutoff is not None:
            if row.get("result_available_at") is None:
                raise ValueError("Training with as_of requires result_available_at")
            available, kickoff = _time(row["result_available_at"]), _time(row["kickoff"])
            if available < kickoff:
                raise ValueError("Result availability precedes kickoff")
            if available >= cutoff or kickoff >= cutoff:
                raise ValueError("Training result is unavailable at as_of")
    return selected


def _summary(rows: Sequence[Row], as_of: Any) -> dict[str, Any]:
    return {"training_rows": len(rows), "training_as_of": _time(as_of).isoformat() if as_of is not None else None,
            "availability_validation": "explicit_result_available_at" if as_of is not None else "caller_responsibility"}


class LeagueFrequency:
    """League H/D/A counts with a fixed symmetric Dirichlet prior."""

    version = "league-frequency-1.0"
    input_fields = ("competition_id", "home_goals", "away_goals")

    def __init__(self, alpha: float = 1.0):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = float(alpha)
        self.counts: dict[str, list[int]] = {}
        self.global_counts = [0, 0, 0]
        self.audit: dict[str, Any] = {}

    def fit(self, rows: Sequence[Row], as_of: Any = None) -> LeagueFrequency:
        selected = _training(rows, as_of)
        counts = defaultdict(lambda: [0, 0, 0])
        self.global_counts = [0, 0, 0]
        for row in selected:
            y = _label(row)
            counts[str(row["competition_id"])][y] += 1
            self.global_counts[y] += 1
        self.counts = dict(counts)
        self.audit = _summary(selected, as_of)
        return self

    def predict_proba(self, row: Row) -> Probabilities:
        if not self.audit:
            raise ValueError("Model must be fitted before prediction")
        counts = self.counts.get(str(row["competition_id"]), self.global_counts)
        return _probabilities([v + self.alpha for v in counts])

    def to_state(self) -> dict[str, Any]:
        return {"state_version": STATE_VERSION, "model_version": self.version, "parameters": {"alpha": self.alpha},
                "input_fields": list(self.input_fields), "counts": self.counts, "global_counts": self.global_counts, "audit": self.audit}

    @classmethod
    def from_state(cls, state: Row) -> LeagueFrequency:
        _check_state(state, cls.version)
        model = cls(**state["parameters"])
        model.counts = {str(k): list(v) for k, v in state["counts"].items()}
        model.global_counts = list(state["global_counts"])
        model.audit = dict(state["audit"])
        return model


def _check_state(state: Row, version: str) -> None:
    if state.get("state_version") != STATE_VERSION or state.get("model_version") != version:
        raise ValueError("Unsupported model state/version")


class MarketBaseline:
    """Proportional de-vig of a complete contemporaneous regulation 1X2 quote.

    The adapter groups H/D/A by provider/bookmaker/quote timestamp and supplies
    their effective availability. A final or closing quote without this timestamp
    is deliberately not a model input. Quotes may precede the prediction cutoff,
    subject to ``max_age_seconds``; future quotes are always rejected.
    """

    version = "market-proportional-1.0"
    input_fields = ("odds_home", "odds_draw", "odds_away", "odds_available_at", "prediction_cutoff")

    def __init__(self, max_age_seconds: float | None = 86400.0):
        if max_age_seconds is not None and (not math.isfinite(max_age_seconds) or max_age_seconds < 0):
            raise ValueError("max_age_seconds must be nonnegative or None")
        self.max_age_seconds = max_age_seconds

    def fit(self, rows: Sequence[Row], as_of: Any = None) -> MarketBaseline:
        return self

    def predict_proba(self, row: Row) -> Probabilities:
        missing = [field for field in self.input_fields if row.get(field) is None]
        if missing:
            raise ValueError("Market prediction lacks required inputs: " + ", ".join(missing))
        cutoff = _time(row["prediction_cutoff"])
        available = _time(row["odds_available_at"])
        if available > cutoff:
            raise ValueError("Market quote is unavailable at prediction_cutoff")
        if row.get("kickoff") is not None and cutoff >= _time(row["kickoff"]):
            raise ValueError("Prematch market cutoff must precede kickoff")
        if row.get("odds_market_type", "1x2") != "1x2" or row.get("odds_period", "regulation") != "regulation":
            raise ValueError("Market baseline requires regulation 1X2 quotes")
        if row.get("odds_line") is not None:
            raise ValueError("Market baseline requires an unhandicapped quote")
        source_time = _time(row["odds_source_timestamp"]) if row.get("odds_source_timestamp") is not None else available
        if source_time > cutoff:
            raise ValueError("Market source timestamp is after prediction_cutoff")
        if self.max_age_seconds is not None and (cutoff - min(available, source_time)).total_seconds() > self.max_age_seconds:
            raise ValueError("Market quote exceeds the configured maximum age")
        times = row.get("odds_selection_timestamps")
        if times is not None and (len(times) != 3 or len({_time(t) for t in times}) != 1 or _time(times[0]) > cutoff):
            raise ValueError("H/D/A prices must belong to the same quote timestamp")
        prices = [float(row[k]) for k in ("odds_home", "odds_draw", "odds_away")]
        if any(not math.isfinite(p) or p <= 1 for p in prices):
            raise ValueError("Decimal market prices must be finite and greater than one")
        return _probabilities([1 / p for p in prices])

    def to_state(self) -> dict[str, Any]:
        return {"state_version": STATE_VERSION, "model_version": self.version,
                "parameters": {"max_age_seconds": self.max_age_seconds}, "input_fields": list(self.input_fields)}

    @classmethod
    def from_state(cls, state: Row) -> MarketBaseline:
        _check_state(state, cls.version)
        return cls(**state["parameters"])


def _effective_rho(rho: float, home_rate: float, away_rate: float) -> float:
    # All four low-score correction factors stay strictly positive, including
    # for unseen teams/rates outside the fitted training range.
    low = -0.999999 / max(home_rate, away_rate)
    high = min(0.999999, 0.999999 / (home_rate * away_rate))
    return float(np.clip(rho, low, high))


def score_probabilities(home_rate: float, away_rate: float, rho: float = 0.0,
                        tail_tolerance: float = 1e-12) -> Probabilities:
    """Adapt score support to each rate and renormalize the retained joint mass."""
    if any(not math.isfinite(x) or x <= 0 for x in (home_rate, away_rate)):
        raise ValueError("Poisson rates must be finite and positive")
    if not 0 < tail_tolerance < 1:
        raise ValueError("tail_tolerance must be between zero and one")
    quantile = min(1 - tail_tolerance / 2, np.nextafter(1.0, 0.0))
    size = max(2, int(max(poisson.ppf(quantile, home_rate), poisson.ppf(quantile, away_rate))))
    support = np.arange(size + 1)
    grid = np.outer(poisson.pmf(support, home_rate), poisson.pmf(support, away_rate))
    rho = _effective_rho(rho, home_rate, away_rate)
    grid[0, 0] *= 1 - home_rate * away_rate * rho
    grid[0, 1] *= 1 + home_rate * rho
    grid[1, 0] *= 1 + away_rate * rho
    grid[1, 1] *= 1 - rho
    return _probabilities((np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()))


class PoissonModel:
    """Ridge-regularized league attack/defence Poisson maximum likelihood.

    Each league has home/away log-rate intercepts and shared team attack/defence
    effects. Positive defence effects mean more conceded goals. Unseen teams use
    zero effects; unseen leagues use the training-only pooled goal means. No
    goals, standings, or other post-match fields enter prediction.
    """

    version = "poisson-attack-defence-1.0"
    input_fields = ("competition_id", "home_team_id", "away_team_id", "home_goals", "away_goals")
    use_dc = False

    def __init__(self, regularization: float = 5.0, max_iterations: int = 500,
                 tail_tolerance: float = 1e-12, half_life_days: float | None = None):
        if not math.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if max_iterations < 1 or not 0 < tail_tolerance < 1:
            raise ValueError("Invalid optimizer iteration count or score tail tolerance")
        if half_life_days is not None and (not math.isfinite(half_life_days) or half_life_days <= 0):
            raise ValueError("half_life_days must be finite and positive")
        self.regularization = float(regularization)
        self.max_iterations = int(max_iterations)
        self.tail_tolerance = float(tail_tolerance)
        self.half_life_days = half_life_days
        self.leagues: dict[str, dict[str, Any]] = {}
        self.fallback_rates = [1.0, 1.0]
        self.audit: dict[str, Any] = {}

    def fit(self, rows: Sequence[Row], as_of: Any = None) -> PoissonModel:
        selected = _training(rows, as_of)
        self.fallback_rates = [float((sum(_goals(r)[j] for r in selected) + 1) / (len(selected) + 1)) for j in (0, 1)]
        groups: dict[str, list[Row]] = defaultdict(list)
        for row in selected:
            groups[str(row["competition_id"])].append(row)
        self.leagues = {league: self._fit_league(group, as_of) for league, group in sorted(groups.items())}
        self.audit = {**_summary(selected, as_of), "optimizer": "scipy-SLSQP" if self.use_dc else "scipy-L-BFGS-B", "unknown_team": "zero_attack_defence",
                      "unknown_league": "training_pooled_goal_means", "rate_log_clip": [-6.0, 4.0],
                      "rho_prediction_rule": "clip_to_positive_low_score_factors" if self.use_dc else None}
        return self

    def _fit_league(self, rows: Sequence[Row], as_of: Any) -> dict[str, Any]:
        teams = sorted({str(row[k]) for row in rows for k in ("home_team_id", "away_team_id")})
        index = {team: i for i, team in enumerate(teams)}
        n = len(teams)
        hi = np.array([index[str(r["home_team_id"])] for r in rows])
        ai = np.array([index[str(r["away_team_id"])] for r in rows])
        goals = np.array([_goals(r) for r in rows], dtype=float)
        home, away = goals[:, 0], goals[:, 1]
        weights = np.ones(len(rows))
        if self.half_life_days is not None:
            reference = _time(as_of) if as_of is not None else max(_time(r["kickoff"]) for r in rows)
            ages = np.array([(reference - _time(r["kickoff"])).total_seconds() / 86400 for r in rows])
            weights = np.exp(-math.log(2) * ages / self.half_life_days)
        prior = np.log((goals.sum(axis=0) + 1) / (len(rows) + 1))
        initial = np.zeros(2 + 2 * n + int(self.use_dc))
        initial[:2] = prior

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            attack, defence = theta[2:2+n], theta[2+n:2+2*n]
            logh = theta[0] + attack[hi] + defence[ai]
            loga = theta[1] + attack[ai] + defence[hi]
            lam, mu = np.exp(logh), np.exp(loga)
            dh, da = lam - home, mu - away
            losses = lam - home * logh + gammaln(home + 1) + mu - away * loga + gammaln(away + 1)
            drho = np.zeros(len(rows))
            if self.use_dc:
                rho = theta[-1]
                if (np.any(1 - lam * mu * rho <= 0) or np.any(1 + lam * rho <= 0)
                        or np.any(1 + mu * rho <= 0) or 1 - rho <= 0):
                    return 1e20, np.zeros_like(theta)
                tau, th, ta, tr = np.ones(len(rows)), np.zeros(len(rows)), np.zeros(len(rows)), np.zeros(len(rows))
                mask = (home == 0) & (away == 0)
                tau[mask] = 1 - lam[mask] * mu[mask] * rho
                th[mask] = ta[mask] = -lam[mask] * mu[mask] * rho
                tr[mask] = -lam[mask] * mu[mask]
                mask = (home == 0) & (away == 1)
                tau[mask] = 1 + lam[mask] * rho
                th[mask], tr[mask] = lam[mask] * rho, lam[mask]
                mask = (home == 1) & (away == 0)
                tau[mask] = 1 + mu[mask] * rho
                ta[mask], tr[mask] = mu[mask] * rho, mu[mask]
                mask = (home == 1) & (away == 1)
                tau[mask], tr[mask] = 1 - rho, -1
                if np.any(tau <= 0):
                    return 1e20, np.zeros_like(theta)
                losses -= np.log(tau)
                dh, da, drho = dh - th / tau, da - ta / tau, -tr / tau
            dh, da = weights * dh, weights * da
            grad = np.zeros_like(theta)
            grad[:2] = [dh.sum(), da.sum()]
            grad[2:2+n] = np.bincount(hi, weights=dh, minlength=n) + np.bincount(ai, weights=da, minlength=n)
            grad[2+n:2+2*n] = np.bincount(ai, weights=dh, minlength=n) + np.bincount(hi, weights=da, minlength=n)
            effects = theta[2:2+2*n]
            loss = float(weights @ losses + 0.5 * self.regularization * (effects @ effects) + 0.5 * np.sum((theta[:2] - prior) ** 2))
            grad[2:2+2*n] += self.regularization * effects
            grad[:2] += theta[:2] - prior
            if self.use_dc:
                loss += self.regularization * theta[-1] ** 2 / 2
                grad[-1] = weights @ drho + self.regularization * theta[-1]
            return loss, grad

        bounds = [(-3.0, 2.0)] * 2 + [(-1.5, 1.5)] * (2*n) + ([(-0.2, 0.2)] if self.use_dc else [])
        if self.use_dc:
            def positive_factors(theta: np.ndarray) -> np.ndarray:
                attack, defence = theta[2:2+n], theta[2+n:2+2*n]
                lam = np.exp(theta[0] + attack[hi] + defence[ai])
                mu = np.exp(theta[1] + attack[ai] + defence[hi])
                rho = theta[-1]
                return np.concatenate((1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, [1 - rho])) - 1e-7
            result = minimize(objective, initial, method="SLSQP", jac=True, bounds=bounds,
                              constraints=[{"type": "ineq", "fun": positive_factors}],
                              options={"maxiter": self.max_iterations, "ftol": 1e-9})
            if np.min(positive_factors(result.x)) < -1e-6:
                raise ValueError("Dixon-Coles optimizer returned invalid low-score factors")
        else:
            result = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                              options={"maxiter": self.max_iterations, "ftol": 1e-10, "gtol": 1e-6})
        if not np.all(np.isfinite(result.x)) or not math.isfinite(float(result.fun)):
            raise ValueError("Poisson fitting produced a nonfinite result")
        if not result.success:
            raise ValueError("Goal-model optimizer did not converge: " + str(result.message))
        theta = result.x
        return {"teams": teams, "home_log_mean": float(theta[0]), "away_log_mean": float(theta[1]),
                "attack": theta[2:2+n].tolist(), "defence": theta[2+n:2+2*n].tolist(),
                "rho": float(theta[-1]) if self.use_dc else 0.0,
                "training_rows": len(rows), "effective_weight": float(weights.sum()),
                "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                "iterations": int(result.nit), "objective": float(result.fun)}

    def rates(self, row: Row) -> tuple[float, float]:
        if not self.audit:
            raise ValueError("Model must be fitted before prediction")
        state = self.leagues.get(str(row["competition_id"]))
        if state is None:
            return tuple(self.fallback_rates)
        index = {team: i for i, team in enumerate(state["teams"])}
        def effect(team: Any, field: str) -> float:
            return state[field][index[str(team)]] if str(team) in index else 0.0
        home, away = row["home_team_id"], row["away_team_id"]
        return (math.exp(float(np.clip(state["home_log_mean"] + effect(home, "attack") + effect(away, "defence"), -6, 4))),
                math.exp(float(np.clip(state["away_log_mean"] + effect(away, "attack") + effect(home, "defence"), -6, 4))))

    def predict_proba(self, row: Row) -> Probabilities:
        rates = self.rates(row)
        rho = self.leagues.get(str(row["competition_id"]), {}).get("rho", 0.0)
        return score_probabilities(*rates, rho=rho, tail_tolerance=self.tail_tolerance)

    def to_state(self) -> dict[str, Any]:
        return {"state_version": STATE_VERSION, "model_version": self.version,
                "parameters": {"regularization": self.regularization, "max_iterations": self.max_iterations,
                               "tail_tolerance": self.tail_tolerance, "half_life_days": self.half_life_days},
                "input_fields": list(self.input_fields), "leagues": self.leagues,
                "fallback_rates": self.fallback_rates, "audit": self.audit}

    @classmethod
    def from_state(cls, state: Row) -> PoissonModel:
        _check_state(state, cls.version)
        model = cls(**state["parameters"])
        model.leagues = dict(state["leagues"])
        model.fallback_rates = list(state["fallback_rates"])
        model.audit = dict(state["audit"])
        return model


class DixonColesModel(PoissonModel):
    """Poisson attack/defence with a fitted low-score dependence correction."""

    version = "dixon-coles-1.0"
    use_dc = True


class EloMapping:
    """Davidson three-way mapping of existing feature-task Elo ratings.

    Ratings are input features, never updated here. ``fit`` estimates the home
    advantage and draw parameter on training labels only, with shrinkage towards
    constructor defaults. Missing or future ratings are rejected.
    """

    version = "elo-davidson-mapping-1.0"
    input_fields = ("home_elo", "away_elo")

    def __init__(self, home_advantage: float = 65.0, draw_parameter: float = 0.7,
                 rating_scale: float = 400.0, regularization: float = 1.0):
        if any(not math.isfinite(x) for x in (home_advantage, draw_parameter, rating_scale, regularization)) or min(draw_parameter, rating_scale, regularization) <= 0:
            raise ValueError("Invalid Davidson mapping parameters")
        self.home_advantage = float(home_advantage)
        self.draw_parameter = float(draw_parameter)
        self.prior_home_advantage = float(home_advantage)
        self.prior_draw_parameter = float(draw_parameter)
        self.rating_scale = float(rating_scale)
        self.regularization = float(regularization)
        self.audit: dict[str, Any] = {"fitted": False, "rating_owner": "features"}

    def _difference(self, row: Row) -> float:
        values = [row.get("home_elo"), row.get("away_elo")]
        if any(v is None or not math.isfinite(float(v)) for v in values):
            raise ValueError("Existing finite home/away Elo features are required")
        if row.get("elo_available_at") is not None:
            if row.get("prediction_cutoff") is None or _time(row["elo_available_at"]) > _time(row["prediction_cutoff"]):
                raise ValueError("Elo features are unavailable at prediction cutoff")
        return float(values[0]) - float(values[1])

    def fit(self, rows: Sequence[Row], as_of: Any = None) -> EloMapping:
        selected = _training(rows, as_of)
        differences = np.array([self._difference(row) for row in selected])
        labels = np.array([_label(row) for row in selected])
        initial = np.array([self.prior_home_advantage / self.rating_scale, math.log(self.prior_draw_parameter)])
        def objective(theta: np.ndarray) -> float:
            log_strength = math.log(10) * (differences / self.rating_scale + theta[0]) / 2
            logits = np.column_stack((log_strength, np.full(len(selected), theta[1]), -log_strength))
            probabilities = softmax(logits, axis=1)
            return float(-np.log(probabilities[np.arange(len(selected)), labels]).sum() + self.regularization * np.sum((theta - initial) ** 2) / 2)
        fitted = minimize(objective, initial, method="L-BFGS-B", bounds=[(-1.5, 1.5), (-5, 3)])
        if not fitted.success or not np.all(np.isfinite(fitted.x)):
            raise ValueError("Elo mapping optimizer did not converge: " + str(fitted.message))
        self.home_advantage = float(fitted.x[0] * self.rating_scale)
        self.draw_parameter = float(math.exp(fitted.x[1]))
        self.audit = {**_summary(selected, as_of), "fitted": True, "rating_owner": "features", "optimizer_success": bool(fitted.success)}
        return self

    def predict_proba(self, row: Row) -> Probabilities:
        strength = math.log(10) * (self._difference(row) + self.home_advantage) / (2 * self.rating_scale)
        return _probabilities(softmax([strength, math.log(self.draw_parameter), -strength]))

    def to_state(self) -> dict[str, Any]:
        return {"state_version": STATE_VERSION, "model_version": self.version,
                "parameters": {"home_advantage": self.prior_home_advantage, "draw_parameter": self.prior_draw_parameter,
                               "rating_scale": self.rating_scale, "regularization": self.regularization},
                "fitted_parameters": {"home_advantage": self.home_advantage, "draw_parameter": self.draw_parameter},
                "input_fields": list(self.input_fields), "audit": self.audit}

    @classmethod
    def from_state(cls, state: Row) -> EloMapping:
        _check_state(state, cls.version)
        model = cls(**state["parameters"])
        model.home_advantage = float(state["fitted_parameters"]["home_advantage"])
        model.draw_parameter = float(state["fitted_parameters"]["draw_parameter"])
        model.audit = dict(state["audit"])
        return model


class LogisticBaseline:
    """Training-only median imputation, missing indicators and standardization.

    Feature names must be an explicit pre-match allowlist supplied by the caller.
    JSON stores the fitted scaler and coefficients, without pickled estimators.
    """

    version = "multinomial-logistic-1.0"
    tree = False

    def __init__(self, feature_names: Sequence[str] = ("home_elo", "away_elo"), regularization: float = 1.0,
                 random_state: int = 1729, max_depth: int = 3, min_samples_leaf: int = 5):
        forbidden = {"home_goals", "away_goals", "outcome", "result", "final_rank", "result_available_at"}
        if not feature_names or len(set(feature_names)) != len(feature_names) or forbidden.intersection(feature_names):
            raise ValueError("Supply distinct pre-match numeric feature names, excluding outcome fields")
        if not math.isfinite(regularization) or regularization <= 0 or max_depth < 1 or min_samples_leaf < 1:
            raise ValueError("Invalid learner parameters")
        self.feature_names = list(feature_names)
        self.input_fields = tuple(feature_names)
        self.regularization = float(regularization)
        self.random_state = int(random_state)
        self.max_depth, self.min_samples_leaf = int(max_depth), int(min_samples_leaf)
        self.fitted: dict[str, Any] = {}
        self.audit: dict[str, Any] = {}

    def _matrix(self, rows: Sequence[Row], fit: bool = False) -> np.ndarray:
        for row in rows:
            if row.get("features_available_at") is not None:
                if row.get("prediction_cutoff") is None or _time(row["features_available_at"]) > _time(row["prediction_cutoff"]):
                    raise ValueError("Numeric features are unavailable at prediction cutoff")
        values = np.array([[float(row[name]) if row.get(name) is not None else np.nan for name in self.feature_names] for row in rows])
        if np.any(np.isinf(values)):
            raise ValueError("Infinite numeric features are not supported")
        missing = np.isnan(values)
        if fit:
            if np.any(missing.all(axis=0)):
                raise ValueError("Each configured feature needs an observed training value")
            self.fitted["median"] = np.nanmedian(values, axis=0).tolist()
        values = np.where(missing, np.asarray(self.fitted["median"]), values)
        values = np.column_stack((values, missing.astype(float)))
        if fit:
            self.fitted["mean"] = values.mean(axis=0).tolist()
            self.fitted["scale"] = np.where(values.std(axis=0) > 0, values.std(axis=0), 1).tolist()
        return (values - np.asarray(self.fitted["mean"])) / np.asarray(self.fitted["scale"])

    def fit(self, rows: Sequence[Row], as_of: Any = None) -> LogisticBaseline:
        selected = _training(rows, as_of)
        labels = np.array([_label(row) for row in selected])
        if set(labels) != {0, 1, 2}:
            raise ValueError("All H/D/A classes must occur in the training window")
        self.fitted = {}
        values = self._matrix(selected, fit=True)
        if self.tree:
            from sklearn.tree import DecisionTreeClassifier
            estimator = DecisionTreeClassifier(criterion="log_loss", max_depth=self.max_depth,
                                               min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)
            estimator.fit(values, labels)
            structure = estimator.tree_
            # sklearn versions differ on whether value stores counts or fractions.
            raw = structure.value[:, 0, :]
            counts = raw / raw.sum(axis=1, keepdims=True) * structure.weighted_n_node_samples[:, None]
            self.fitted["learner"] = {"children_left": structure.children_left.tolist(), "children_right": structure.children_right.tolist(),
                                      "feature": structure.feature.tolist(), "threshold": structure.threshold.tolist(), "counts": counts.tolist()}
        else:
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(C=1 / self.regularization, max_iter=1000, solver="lbfgs", random_state=self.random_state)
            estimator.fit(values, labels)
            if np.any(estimator.n_iter_ >= estimator.max_iter):
                raise ValueError("Logistic optimizer exhausted its iteration budget")
            self.fitted["learner"] = {"coef": estimator.coef_.tolist(), "intercept": estimator.intercept_.tolist()}
        self.audit = {**_summary(selected, as_of), "preprocessing": "training_median_missing_indicators_standard_scale",
                      "classes": ["home", "draw", "away"], "tree_leaf_smoothing_alpha": 1.0 if self.tree else None}
        return self

    def predict_proba(self, row: Row) -> Probabilities:
        if not self.audit:
            raise ValueError("Model must be fitted before prediction")
        values = self._matrix([row])[0]
        learner = self.fitted["learner"]
        if self.tree:
            node = 0
            while learner["children_left"][node] != -1:
                node = learner["children_left"][node] if values[learner["feature"][node]] <= learner["threshold"][node] else learner["children_right"][node]
            return _probabilities(np.asarray(learner["counts"][node]) + 1.0)
        return _probabilities(softmax(np.asarray(learner["coef"]) @ values + np.asarray(learner["intercept"])))

    def to_state(self) -> dict[str, Any]:
        return {"state_version": STATE_VERSION, "model_version": self.version,
                "parameters": {"feature_names": self.feature_names, "regularization": self.regularization,
                               "random_state": self.random_state, "max_depth": self.max_depth, "min_samples_leaf": self.min_samples_leaf},
                "input_fields": self.feature_names, "fitted": self.fitted, "audit": self.audit}

    @classmethod
    def from_state(cls, state: Row) -> LogisticBaseline:
        _check_state(state, cls.version)
        model = cls(**state["parameters"])
        model.fitted, model.audit = dict(state["fitted"]), dict(state["audit"])
        return model


class TreeBaseline(LogisticBaseline):
    """Small deterministic decision tree with Laplace-smoothed H/D/A leaves."""

    version = "decision-tree-1.0"
    tree = True


MODEL_TYPES = {cls.version: cls for cls in (LeagueFrequency, MarketBaseline, PoissonModel, DixonColesModel, EloMapping, LogisticBaseline, TreeBaseline)}


def model_from_state(state: Row) -> Any:
    """Restore one of the explicitly registered model classes from JSON data."""
    version = state.get("model_version")
    if version not in MODEL_TYPES:
        raise ValueError("Unknown model version")
    return MODEL_TYPES[version].from_state(state)
