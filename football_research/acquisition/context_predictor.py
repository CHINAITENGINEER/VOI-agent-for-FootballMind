"""Small market-offset outcome model for the shared visible-state interface.

Only declared, already-visible text enters a training-only TF-IDF vocabulary.
Every physical fixture has equal objective/encoder weight, regardless of how
many runs or prefixes it supplied. This model is not a query-value estimator.
The real raw/identity/settlement gateway remains the caller's responsibility.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.feature_extraction.text import TfidfVectorizer

from .state import parse_time
from .text_returns import _p, _canonical
from .visible_state import VERSION as STATE_VERSION

VERSION = "market-offset-visible-context-v1"
ENCODER_VERSION = "visible-character-tfidf-context-v1"
PARAMETERS = {"regularization": 10., "max_features": 512, "min_physical_matches": 12,
              "min_class_matches": 1, "ngram_min": 3, "ngram_max": 5}
NUMERIC = ("body_fraction", "headline_fraction", "link_fraction", "failure_fraction", "no_data_fraction",
           "unknown_attempt_fraction", "unknown_document_fraction", "last_failure", "last_no_data",
           "last_unknown", "previous_failure", "previous_no_data", "previous_unknown",
           "publication_missing_fraction", "capture_age", "publication_age")
ORIGINS = {"real_observational", "constructed_mechanism"}


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _grams(texts):
    # No n-grams cross document/label boundaries. No URL or entity ID tokens.
    for text in texts:
        text = " ".join(text.lower().split())
        for n in range(3, 6):
            for i in range(len(text)-n+1):
                yield text[i:i+n]


def _state(value, *, allow_missing_prior=False):
    if not isinstance(value, dict) or value.get("schema_version") != STATE_VERSION:
        raise ValueError("A versioned encode_visible_state result is required")
    before = value["before"]
    now, cutoff = parse_time(before["decision_at"]), parse_time(before["prediction_cutoff"])
    kickoff = parse_time(before["fixture"]["kickoff"])
    if not now <= cutoff < kickoff:
        raise ValueError("Visible prediction must be available before its prematch cutoff")
    if allow_missing_prior and before["prior_probabilities"] is None:
        if before.get("prior_basis") != "not_supplied_no_prior_inferred_from_current_prediction":
            raise ValueError("Missing prior requires the explicit no-prior visible-state basis")
        p0 = None
    else:
        p0 = _p(before["prior_probabilities"])
        if before.get("prior_basis") not in {"observed_market_snapshot", "explicit_open_text_initial_prior"}:
            raise ValueError("Explicit original prior required, not a current posterior")
    content = {}
    for entry in before.get("contents", []):
        text = entry.get("text")
        if not isinstance(text, str) or entry.get("content_id") != "text-"+sha256(text.encode()).hexdigest():
            raise ValueError("Visible content identity differs from its actual text")
        content[entry["content_id"]] = text
    documents, parts, sources, captures, published = {}, defaultdict(set), set(), [], []
    for doc in before.get("documents", []):
        if doc["document_index"] in documents:
            raise ValueError("Visible document indices must be unique")
        observed, available = parse_time(doc["observed_at"]), parse_time(doc["available_at"])
        host = urlsplit(doc["url"]).hostname
        if (not host or not doc.get("document_id") or not re.fullmatch(r"[0-9a-f]{64}", doc.get("raw_sha256", ""))
                or not observed <= available <= now):
            raise ValueError("Visible document source/clock binding is invalid")
        timestamps = [parse_time(doc[key]) for key in ("source_published_at", "source_modified_at") if doc.get(key) is not None]
        if any(stamp > observed for stamp in timestamps):
            raise ValueError("Source publication/modification follows capture")
        documents[doc["document_index"]] = doc
        captures.append((now-available).total_seconds()/86400.)
        pub = parse_time(doc["source_published_at"]) if doc.get("source_published_at") is not None else None
        published.append(None if pub is None else (now-pub).total_seconds()/86400.)
        for ref in doc.get("content_refs", []):
            role, text = ref.get("role"), content.get(ref.get("content_id"))
            if (doc.get("status") != "OK" or doc.get("source_role") not in {"article", "liveblog"}
                    or role not in {"body", "headline"} or text is None
                    or type(ref.get("start")) is not int or type(ref.get("end")) is not int
                    or ref["start"] < 0 or ref["end"]-ref["start"] != len(text)):
                raise ValueError("Text must have a declared visible article span")
            if any(parse_time(ref[k]) > observed for k in ("source_published_at", "source_modified_at") if ref.get(k) is not None):
                raise ValueError("Visible paragraph is from after actual capture")
            if text.strip():
                parts[role].add(text)
                sources.add(host.lower())
    for link in before.get("observed_links", []):
        doc = documents.get(link.get("document_index"))
        if doc is None or doc.get("status") != "OK" or link["link_ref"] not in doc.get("observed_link_refs", []):
            raise ValueError("Observed link must belong to an already-read parent")
        for anchor in link.get("observed_anchors", []):
            text = anchor.get("text")
            if not isinstance(text, str):
                raise ValueError("Observed link label must be actual text")
            if text.strip():
                parts["link"].add(text)
                sources.add(urlsplit(doc["url"]).hostname.lower())
    texts = sorted(set().union(*parts.values())) if parts else []
    statuses = [step.get("status") for step in before.get("history", {}).get("attempts", [])]
    def fraction(items, predicate):
        return sum(predicate(x) for x in items)/max(1, len(items))
    def age(values):
        values = [v for v in values if v is not None]
        return 0. if not values else float(np.mean([math.log1p(v)/(1.+math.log1p(v)) for v in values]))
    n = max(1, sum(len(parts[k]) for k in ("body", "headline", "link")))
    numeric = [len(parts[k])/n for k in ("body", "headline", "link")]
    numeric += [fraction(statuses, lambda s: s == "FAILED"), fraction(statuses, lambda s: s == "NO_DATA"),
                fraction(statuses, lambda s: s not in {"FAILED", "NO_DATA", "OK"}),
                fraction(list(documents.values()), lambda d: d.get("source_role") == "unknown")]
    for offset in (1, 2):
        status = statuses[-offset] if len(statuses) >= offset else None
        numeric += [float(status == "FAILED"), float(status == "NO_DATA"),
                    float(status is not None and status not in {"FAILED", "NO_DATA", "OK"})]
    numeric += [fraction(published, lambda p: p is None), age(captures), age(published)]
    return {"provider_match_id": before["match_id"], "decision_at": now.isoformat(),
            "prediction_cutoff": cutoff.isoformat(), "kickoff_at": kickoff.isoformat(),
            "prior": p0, "texts": texts, "sources": sorted(sources), "numeric": numeric}


def _training(records, as_of, evidence_origin, *, allow_missing_prior=False):
    if evidence_origin not in ORIGINS or not isinstance(records, (list, tuple)):
        raise ValueError("Explicit origin and raw visible-state prefix records required")
    at, facts, aliases, unique = parse_time(as_of), {}, {}, {}
    for row in records:
        physical, y = row.get("physical_match_id"), row.get("outcome")
        if not isinstance(physical, str) or not physical or type(y) is not int or y not in (0, 1, 2):
            raise ValueError("Canonical physical identity and independent three-class outcome required")
        if row.get("purpose", "f_train") != "f_train":
            raise ValueError("Only f_train fixtures may fit the predictor")
        state, label_at = _state(row["visible_state"], allow_missing_prior=allow_missing_prior), parse_time(row["result_available_at"])
        if not parse_time(state["kickoff_at"]) <= label_at <= at:
            raise ValueError("Settlement must follow kickoff and precede training cutoff")
        fact = (y, state["kickoff_at"])
        if physical in facts and facts[physical] != fact:
            raise ValueError("Conflicting outcome or kickoff for one physical fixture")
        facts[physical] = fact
        provider = state["provider_match_id"]
        if provider in aliases and aliases[provider] != physical:
            raise ValueError("One provider identity cannot represent different physical fixtures")
        aliases[provider] = physical
        # Exact feature-equivalent repeated observations do not change weights.
        key = _canonical([physical, state["prior"], state["texts"], state["sources"], state["numeric"]])
        item = {"physical_match_id": physical, "outcome": y, "result_available_at": label_at.isoformat(), "state": state}
        if key not in unique or _canonical(item) < _canonical(unique[key]):
            unique[key] = item
    rows = [unique[k] for k in sorted(unique)]
    counts = Counter(fact[0] for fact in facts.values())
    reasons = []
    if len(facts) < PARAMETERS["min_physical_matches"]:
        reasons.append("fewer_than_12_independent_physical_matches")
    if any(counts[y] < PARAMETERS["min_class_matches"] for y in range(3)):
        reasons.append("physical_matches_do_not_cover_all_three_outcomes")
    report = {"ready_to_fit": not reasons, "reasons": reasons, "physical_matches": len(facts),
        "class_counts": {str(y): counts[y] for y in range(3)}, "input_records": len(records),
        "distinct_prefixes": len(rows), "evidence_origin": evidence_origin,
        "gate_scope": "necessary_global_gate_not_predictive_quality_or_local_support"}
    return rows, aliases, report


def training_readiness(records, *, as_of, evidence_origin):
    return _training(records, as_of, evidence_origin)[2]


class ContextPredictor:
    version = VERSION

    def __init__(self):
        self.metadata, self.encoder, self.coefficients = {}, {}, []
        self.model_version = ""

    def _vector(self, state):
        counts = Counter(_grams(state["texts"]))
        vocabulary = self.encoder["vocabulary"]
        lexical = np.asarray([(1.+math.log(counts[token])) if counts[token] else 0. for token in vocabulary])
        lexical *= np.asarray(self.encoder["idf"])
        norm = np.linalg.norm(lexical)
        if norm == 0:
            return np.zeros(len(vocabulary)+len(NUMERIC)+len(self.encoder["sources"])), 0
        lexical /= norm
        categories = np.asarray([float(source in state["sources"]) for source in self.encoder["sources"]])
        if np.linalg.norm(categories):
            categories /= np.linalg.norm(categories)
        numeric = np.asarray(state["numeric"])/math.sqrt(len(NUMERIC))
        # Bounded blocks; no unpenalized intercept or free frequency recalibration.
        return np.concatenate([lexical, numeric, categories]), len(set(counts).intersection(vocabulary))

    def _fit_rows(self, rows):
        """Shared encoder/regularized offset optimizer; rows have explicit offsets.

        The original v1 caller supplies its unchanged observed-prior rows.
        Optional-prior callers must register their learned offset separately.
        """
        per_match = defaultdict(set)
        for row in rows:
            per_match[row["physical_match_id"]].update(row["state"]["texts"])
        corpus = [sorted(per_match[mid]) for mid in sorted(per_match)]
        if not any(any(_grams(texts)) for texts in corpus):
            raise ValueError("Context predictor not_ready: no_visible_text_vocabulary")
        vectorizer = TfidfVectorizer(analyzer=_grams, max_features=PARAMETERS["max_features"], sublinear_tf=True)
        vectorizer.fit(corpus)  # One document per physical game, not per prefix.
        self.encoder = {"version": ENCODER_VERSION, "vocabulary": vectorizer.get_feature_names_out().tolist(),
            "idf": vectorizer.idf_.tolist(), "numeric_names": list(NUMERIC),
            "sources": sorted({source for row in rows for source in row["state"]["sources"]})}
        x = np.asarray([self._vector(row["state"])[0] for row in rows])
        labels = np.asarray([r["outcome"] for r in rows])
        offsets = np.log(np.asarray([r["state"]["prior"] for r in rows]))
        sizes = Counter(r["physical_match_id"] for r in rows)
        weights = np.asarray([1./(len(sizes)*sizes[r["physical_match_id"]]) for r in rows])
        def objective(flat):
            coef = flat.reshape(x.shape[1], 3)
            logits = offsets+x@coef
            loss = weights@(logsumexp(logits, axis=1)-logits[np.arange(len(labels)), labels])
            difference = softmax(logits, axis=1)
            difference[np.arange(len(labels)), labels] -= 1.
            lam = PARAMETERS["regularization"]
            return float(loss+lam/2*np.sum(coef*coef)), (x.T@(difference*weights[:, None])+lam*coef).ravel()
        result = minimize(objective, np.zeros(x.shape[1]*3), jac=True, method="L-BFGS-B",
                          options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9})
        if not result.success or not np.all(np.isfinite(result.x)):
            self.encoder = {}
            raise ValueError("Context predictor optimizer did not converge")
        self.coefficients = result.x.reshape(x.shape[1], 3).tolist()
        return sizes, result

    def fit(self, records, *, as_of, dataset_version, evidence_origin):
        rows, aliases, ready = _training(records, as_of, evidence_origin)
        if not ready["ready_to_fit"]:
            raise ValueError("Context predictor not_ready: "+", ".join(ready["reasons"]))
        if not isinstance(dataset_version, str) or not dataset_version.strip():
            raise ValueError("Explicit input dataset version required")
        sizes, result = self._fit_rows(rows)
        self.metadata = {"training_as_of": parse_time(as_of).isoformat(),
            "fit_execution_at": datetime.now(timezone.utc).isoformat(), "dataset_version": dataset_version,
            "evidence_origin": evidence_origin, "visible_state_schema": STATE_VERSION,
            "training_physical_match_ids": sorted(sizes), "training_match_ids": sorted(aliases),
            "training_aliases": aliases, "readiness": ready, "training_records_sha256": _digest(rows),
            "match_weighting": "equal_physical_fixture_then_equal_distinct_visible_prefix",
            "encoder_weighting": "one_union_of_visible_texts_per_training_physical_fixture",
            "input_scope": "before_only_no_candidate_return_current_posterior_identifier_or_cost_features",
            "history_summary": "typed_status_fractions_and_last_two_statuses_full_history_retained_upstream",
            "support_scope": "global_sample_gate_and_known_character_features_not_conditional_calibration",
            "label_scope": "caller_verified_raw_gateway_required_not_proven_by_outcome_integer",
            "empty_context": "exact_market_prior_without_fitted_intercept",
            "calibration_status": "not_established", "production_ready": False,
            "optimizer": {"method": "scipy-L-BFGS-B", "iterations": int(result.nit), "objective": float(result.fun)}}
        self.model_version = VERSION+":"+_digest(self._payload())[:24]
        return self

    def predict(self, visible_state, *, physical_match_id):
        self._check()
        state = _state(visible_state)
        if not isinstance(physical_match_id, str) or not physical_match_id:
            raise ValueError("A canonical physical prediction identity is required")
        if (physical_match_id in self.metadata["training_physical_match_ids"]
                or state["provider_match_id"] in self.metadata["training_match_ids"]):
            raise ValueError("Training physical fixtures/provider aliases cannot be held-out predictions")
        if parse_time(state["decision_at"]) <= parse_time(self.metadata["training_as_of"]):
            raise ValueError("Prediction decision must follow the frozen training cutoff")
        vector, known = self._vector(state)
        reason = "no_visible_text" if not state["texts"] else "all_text_out_of_vocabulary" if not known else None
        unknown_sources = sorted(set(state["sources"])-set(self.encoder["sources"]))
        residual = vector@np.asarray(self.coefficients)
        probabilities = list(state["prior"]) if reason else softmax(np.log(state["prior"])+residual).tolist()
        return {"schema": VERSION, "model_version": self.model_version, "match_id": state["provider_match_id"],
            "physical_match_id": physical_match_id, "probabilities": probabilities,
            "prior_probabilities": state["prior"], "logit_residual": residual.tolist(),
            "prediction_kind": "market_prior_fallback" if reason else "regularized_context_forecast",
            "fallback_reason": reason, "known_character_features": known, "unknown_sources": unknown_sources,
            "support_status": "prior_only" if reason else "source_extrapolation" if unknown_sources else "shared_feature_projection",
            "evidence_origin": self.metadata["evidence_origin"], "calibration_status": "not_established",
            "execution_scope": "research_prediction_not_live_deployment_authorization", "production_ready": False}

    def _payload(self):
        return {"schema": VERSION, "parameters": PARAMETERS, "metadata": self.metadata,
                "encoder": self.encoder, "coefficients": self.coefficients}

    def _check(self):
        if not self.model_version or self.model_version != VERSION+":"+_digest(self._payload())[:24]:
            raise ValueError("Frozen context predictor content/version mismatch")

    def save(self, path):
        self._check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({**self._payload(), "model_version": self.model_version},
                                   ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != VERSION or payload.get("parameters") != PARAMETERS:
            raise ValueError("Unsupported context predictor schema or fixed parameters")
        model = cls()
        model.metadata, model.encoder, model.coefficients = [payload[k] for k in ("metadata", "encoder", "coefficients")]
        model.model_version = payload.get("model_version")
        model._check()
        enc, meta = model.encoder, model.metadata
        features = len(enc["vocabulary"])+len(NUMERIC)+len(enc["sources"])
        if (enc.get("version") != ENCODER_VERSION or enc.get("numeric_names") != list(NUMERIC)
                or not 1 <= len(enc["vocabulary"]) <= PARAMETERS["max_features"]
                or enc["vocabulary"] != sorted(set(enc["vocabulary"])) or enc["sources"] != sorted(set(enc["sources"]))
                or len(enc["idf"]) != len(enc["vocabulary"]) or any(not math.isfinite(x) or x < 1 for x in enc["idf"])
                or np.asarray(model.coefficients).shape != (features, 3) or not np.all(np.isfinite(model.coefficients))
                or meta.get("evidence_origin") not in ORIGINS or meta.get("visible_state_schema") != STATE_VERSION
                or len(set(meta["training_physical_match_ids"])) < PARAMETERS["min_physical_matches"]
                or any(meta["readiness"]["class_counts"][str(y)] < 1 for y in range(3))):
            raise ValueError("Invalid frozen context encoder, parameters or training support")
        return model


__all__ = ["ContextPredictor", "training_readiness", "VERSION", "PARAMETERS"]
