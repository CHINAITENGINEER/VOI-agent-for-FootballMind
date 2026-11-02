"""Offline, f-bound inference for the saved full-text/state research selectors.

This entry point has no network, extractor, label, training or runtime side effect.
All supplied candidate bodies and their validated semantic reviews are already
visible to g. Only adopted bodies enter f. Missing reviews are never empty claims.
The original pre-query policies and frozen model packages remain independent.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import normalize

from .entity_semantic_f import (
    EntitySemanticF, FEATURES, REPRESENTATION_SHA256, _modeled, _predict, predicate_key,
)
from .semantic_f import validate_response

SCHEMA = "postread-paired-selector-v1"
STATE_NAMES = ["p_home", "p_draw", "p_away", "entropy_nats", "top_two_margin",
               "adopted_body_count", "remaining_body_count", "hours_to_kickoff", "time_unknown"] + [
                   "state_title_entities:" + x for x in FEATURES] + ["state_body_entities:" + x for x in FEATURES]
SEMANTIC_NAMES = ["candidate_claim_count", "candidate_unmodeled_count", "repeated_predicates",
                  "coarse_overlap_count", "candidate_status_conflicts"] + ["candidate_entity_delta:" + x for x in FEATURES]
FEATURE_FIELDS = {"sample_id", "physical_match_id", "split", "candidate_text", "candidate_title",
                  "adopted_text", "state_numeric", "state_feature_names", "semantic_numeric", "semantic_feature_names"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1) or abs(p.sum()-1) > 1e-10:
        raise ValueError("Three finite, strictly positive, normalized probabilities required")
    return p


def project_claims(prior, titles, bodies):
    """Exact EntityEvidenceBank set/count rule; timestamps are not synthesized."""
    _probabilities(prior)
    title_keys = {predicate_key(c) for c in titles if _modeled(c)}
    body_keys = {predicate_key(c) for c in bodies if _modeled(c)} - title_keys
    def counts(keys):
        ct = Counter(":".join((t[0], t[2], t[3])) for t in keys)
        return [float(ct[k]) for k in FEATURES]
    tc, bc = counts(title_keys), counts(body_keys)
    return dict(prior=list(prior), title=[float(n > 0) for n in tc], novel_body=[float(n > 0) for n in bc],
                title_entity_counts=tc, novel_body_entity_counts=bc, representation_sha256=REPRESENTATION_SHA256)


@dataclass(frozen=True)
class BodyCandidate:
    candidate_id: str
    action_id: str
    body_sha256: str
    semantic_record_json: str
    clocks_json: str = "{}"

    @classmethod
    def from_record(cls, *, candidate_id, action_id, body_sha256, record, clocks=None):
        if not candidate_id or not action_id:
            raise ValueError("Explicit candidate_id and action_id required")
        if record["input"]["channel"] != "body":
            raise ValueError("An acquired full-body semantic input is required")
        validate_response(record["input"], record["response"])
        if text_sha256(record["input"]["text"]) != body_sha256:
            raise ValueError("Body SHA differs from reviewed semantic input")
        # Retain only the input/response pair, never upstream targets or metrics.
        pair = {k: record[k] for k in ("input", "response")}
        return cls(candidate_id, action_id, body_sha256, json.dumps(pair, ensure_ascii=False),
                   json.dumps(clocks or {}, ensure_ascii=False))

    @property
    def record(self):
        return json.loads(self.semantic_record_json)

    @property
    def input_id(self):
        return self.record["input"]["input_id"]


class PostreadSelector:
    """Load an explicitly trusted local sklearn package and its exact paired f."""
    def __init__(self, model_dir, f_path):
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        if metadata.get("schema") != "unified-f-postread-g-experiment-v1":
            raise ValueError("Unsupported postread model schema")
        self.f_sha256 = sha256(f_path)
        if metadata.get("f_model_sha256") != self.f_sha256:
            raise ValueError("Paired f SHA mismatch; cannot use another f with this g")
        model_path = model_dir / metadata["model_file"]
        if model_path.resolve().parent != model_dir.resolve():
            raise ValueError("Model file must be inside the trusted model directory")
        self.g_sha256 = sha256(model_path)
        if self.g_sha256 != metadata["model_sha256"]:
            raise ValueError("Saved g model bytes differ from model.json")
        self._f = EntitySemanticF.load(f_path)
        self._coefficients = np.asarray(self._f.coefficients).copy()
        self._coefficients.setflags(write=False)
        self._bundle = joblib.load(model_path)
        if self._bundle["protocol"]["f_model_sha256"] != self.f_sha256:
            raise ValueError("Serialized g training protocol f binding differs")
        for key, names in (("state_feature_names", STATE_NAMES), ("semantic_feature_names", SEMANTIC_NAMES)):
            if metadata[key] != names or self._bundle[key] != names:
                raise ValueError("Saved feature names/order differ from this adapter")
        self.metadata = deepcopy(metadata)
        self.model_names = tuple(self._bundle["fitted"])
        self._embedding_cache = {}

    @property
    def f_model_version(self):
        return self._f.model_version

    def predict_f(self, projection):
        return _predict(self._coefficients, projection).tolist()

    def _encode(self, text, kind):
        key = (kind, text_sha256(text))
        if key not in self._embedding_cache:
            encoder = self._bundle[kind + "_encoder"]
            tf = encoder["vectorizer"].transform([text])
            value = normalize(encoder["svd"].transform(tf), norm="l2")[0]
            self._embedding_cache[key] = value
        return self._embedding_cache[key]

    def vocabulary_coverage(self, text):
        encoder = self._bundle["body_encoder"]["vectorizer"]
        terms = encoder.build_analyzer()(text)
        known = sum(t in encoder.vocabulary_ for t in terms)
        return dict(analyzed_terms=len(terms), known_terms=known,
                    known_fraction=known/len(terms) if terms else None,
                    embedding_norm=float(np.linalg.norm(self._encode(text, "body"))),
                    meaning="lexical vocabulary coverage, not semantic correctness")

    def design(self, row):
        if set(row) != FEATURE_FIELDS:
            raise ValueError("Feature input must contain exactly the pre-decision allowlist")
        if row["state_feature_names"] != STATE_NAMES or row["semantic_feature_names"] != SEMANTIC_NAMES:
            raise ValueError("Feature order differs")
        for key in ("candidate_text", "candidate_title", "adopted_text"):
            if not isinstance(row[key], str):
                raise ValueError("Actual text strings required")
        if not row["candidate_text"].strip():
            raise ValueError("Full candidate body is missing")
        state, sem = np.asarray(row["state_numeric"], dtype=float), np.asarray(row["semantic_numeric"], dtype=float)
        if state.shape != (57,) or sem.shape != (29,) or not np.isfinite(state).all() or not np.isfinite(sem).all():
            raise ValueError("Invalid finite state/semantic feature dimensions")
        _probabilities(state[:3])
        body, adopted = self._encode(row["candidate_text"], "body"), self._encode(row["adopted_text"], "body")
        title = self._encode(row["candidate_title"], "title")
        out = {"state": state, "body": body, "title_local": np.r_[state, title], "semantic": np.r_[state, sem],
               "fulltext_state": np.r_[state, sem, body, adopted, abs(body-adopted), body*adopted]}
        if any(len(v) != self.metadata["feature_dimensions"][k] for k, v in out.items()):
            raise ValueError("Frozen encoder/design dimensions differ")
        return out

    def score(self, row, *, model_name):
        designs = self.design(row)
        spec = self._bundle["fitted"][model_name]
        if spec["kind"] == "zero":
            value = 0.
        elif spec["kind"] == "constant":
            value = self._bundle["target_mean"]
        else:
            xx = spec["scaler"].transform(designs[spec["input_key"]].reshape(1, -1))
            value = float(spec["estimator"].predict(xx)[0])*self._bundle["target_scale"]+self._bundle["target_mean"]
        if not math.isfinite(value):
            raise ValueError("Nonfinite g output")
        return float(value)


class PostreadSession:
    """A local research state; all full bodies here are g-seen, not yet f-adopted.

    Titles supplied as known_title_records are already part of f's starting state.
    Full-body semantic reviews are precomputed for every candidate; review cost
    therefore belongs to up-front processing, not the remaining adoption cost.
    """
    def __init__(self, selector, *, physical_match_id, prior, candidates, known_title_records=(),
                 hours_to_kickoff=None, adopted_ids=(), evidence_order=None, unread_candidate_ids=()):
        self.selector = selector
        self.physical_match_id = physical_match_id
        self.prior = _probabilities(prior).tolist()
        self.candidates = {c.candidate_id: c for c in candidates}
        if len(self.candidates) != len(candidates) or len({c.action_id for c in candidates}) != len(candidates):
            raise ValueError("Candidate and action IDs must be unique")
        self.unread_candidate_ids = tuple(unread_candidate_ids)
        if set(self.unread_candidate_ids) & self.candidates.keys():
            raise ValueError("A body cannot be both read and unread")
        self._title_records = deepcopy(list(known_title_records))
        all_records = [c.record for c in candidates] + self._title_records
        for c in candidates:
            if c.record["input"]["channel"] != "body" or text_sha256(c.record["input"]["text"]) != c.body_sha256:
                raise ValueError("Read candidate body/semantic binding differs")
        for record in all_records:
            validate_response(record["input"], record["response"])
            if record["input"]["fixture"]["physical_match_id"] != physical_match_id:
                raise ValueError("Semantic record belongs to another physical match")
        if any(r["input"]["channel"] != "title" for r in self._title_records):
            raise ValueError("Only title records may enter the initial known-title state")
        self._adopted = set(adopted_ids)
        self.evidence_order = tuple(self.candidates if evidence_order is None else evidence_order)
        if set(self.evidence_order) != self.candidates.keys() or len(self.evidence_order) != len(self.candidates):
            raise ValueError("Evidence order must exactly cover the read menu")
        if not self._adopted <= self.candidates.keys() or len(self._adopted) != len(adopted_ids):
            raise ValueError("Only distinct, g-read bodies can already be adopted")
        if hours_to_kickoff is not None and (not math.isfinite(hours_to_kickoff) or hours_to_kickoff < 0):
            raise ValueError("Invalid pre-match hours_to_kickoff")
        self.hours_to_kickoff = hours_to_kickoff
        self.trace = []

    @property
    def adopted_ids(self):
        return [cid for cid in self.evidence_order if cid in self._adopted]

    def _claims(self):
        titles = [c for r in self._title_records for c in r["response"]["claims"]]
        bodies = [c for cid in self.adopted_ids for c in self.candidates[cid].record["response"]["claims"]]
        return titles, bodies

    def prediction(self):
        titles, bodies = self._claims()
        projection = project_claims(self.prior, titles, bodies)
        return dict(probabilities=self.selector.predict_f(projection), projection=projection,
                    used_input_ids=[r["input"]["input_id"] for r in self._title_records] + [
                        self.candidates[cid].input_id for cid in self.adopted_ids],
                    f_adopted_body_ids=self.adopted_ids, f_model_sha256=self.selector.f_sha256)

    def feature(self, candidate_id, *, sample_id="research", split="inference"):
        if candidate_id not in self.candidates or candidate_id in self._adopted:
            raise ValueError("Candidate must be g-read and not f-adopted")
        candidate = self.candidates[candidate_id].record
        candidate_claims = candidate["response"]["claims"]
        titles, bodies = self._claims()
        before = self.prediction()
        p, projection = before["probabilities"], before["projection"]
        known = {predicate_key(c) for c in titles+bodies if _modeled(c)}
        proposed = {predicate_key(c) for c in candidate_claims if _modeled(c)}
        coarse = {":".join((t[0], t[2], t[3])) for t in known}
        novel = proposed-known
        delta = Counter(":".join((t[0], t[2], t[3])) for t in novel)
        statuses = defaultdict(set)
        for team, entity, kind, value in known | proposed:
            if kind == "availability":
                statuses[team, entity].add(value)
        order = sorted(p, reverse=True)
        state = list(p)+[-sum(x*math.log(x) for x in p), order[0]-order[1],
                         float(len(self._adopted)), float(len(self.candidates)-len(self._adopted)),
                         0. if self.hours_to_kickoff is None else self.hours_to_kickoff,
                         float(self.hours_to_kickoff is None)]
        state += projection["title_entity_counts"]+projection["novel_body_entity_counts"]
        sem = [float(len(candidate_claims)), float(sum(not _modeled(c) for c in candidate_claims)),
               float(len(known & proposed)), float(sum(":".join((t[0], t[2], t[3])) in coarse for t in novel)),
               float(sum(len(v) > 1 for v in statuses.values()))]+[float(delta[k]) for k in FEATURES]
        return dict(sample_id=sample_id, physical_match_id=self.physical_match_id, split=split,
                    candidate_text=candidate["input"]["text"], candidate_title=candidate["input"]["title"],
                    adopted_text="\n\n".join(self.candidates[c].record["input"]["text"] for c in self.adopted_ids),
                    state_numeric=state, state_feature_names=STATE_NAMES,
                    semantic_numeric=sem, semantic_feature_names=SEMANTIC_NAMES)

    def choose(self, *, model_name, remaining_updates, incremental_cost=0.):
        if isinstance(remaining_updates, bool) or not isinstance(remaining_updates, int) or remaining_updates < 0:
            raise ValueError("Nonnegative integer remaining update budget required")
        if not math.isfinite(incremental_cost) or incremental_cost < 0:
            raise ValueError("Finite nonnegative incremental loss-unit cost required")
        rows = [dict(candidate_id=cid, action_id=c.action_id,
                     predicted_gain=self.selector.score(self.feature(cid), model_name=model_name))
                for cid, c in self.candidates.items() if cid not in self._adopted]
        for row in rows:
            row["predicted_net_gain"] = row["predicted_gain"]-incremental_cost
        best = max(rows, key=lambda r: r["predicted_net_gain"]) if rows else None
        reason = "budget_exhausted" if remaining_updates == 0 else (
            "no_unadopted_read_candidate" if best is None else (
                "nonpositive_predicted_net_gain" if best["predicted_net_gain"] <= 0 else "positive_predicted_net_gain"))
        chosen = best if reason == "positive_predicted_net_gain" else None
        return dict(schema=SCHEMA, mode="offline_research", production_selection_allowed=False,
                    model_name=model_name, g_model_sha256=self.selector.g_sha256, f_model_sha256=self.selector.f_sha256,
                    g_read_body_ids=list(self.candidates), f_adopted_body_ids=self.adopted_ids,
                    unread_candidate_ids=list(self.unread_candidate_ids), candidates=rows,
                    selected_candidate_id=chosen["candidate_id"] if chosen else None,
                    selected_action_id=chosen["action_id"] if chosen else None, stop_reason=None if chosen else reason,
                    incremental_cost_loss_units=incremental_cost)

    def adopt(self, candidate_id):
        if candidate_id not in self.candidates or candidate_id in self._adopted:
            raise ValueError("Only a read, not-yet-adopted body may enter f")
        before = self.prediction()
        self._adopted.add(candidate_id)
        try:
            after = self.prediction()
        except Exception:
            self._adopted.remove(candidate_id)
            raise
        candidate = self.candidates[candidate_id]
        update = dict(candidate_id=candidate_id, action_id=candidate.action_id, body_sha256=candidate.body_sha256,
                      source_url=candidate.record["input"]["url"], source_clocks=json.loads(candidate.clocks_json),
                      p_before=before["probabilities"], p_after=after["probabilities"],
                      used_input_ids_before=before["used_input_ids"], used_input_ids_after=after["used_input_ids"],
                      probability_l1=float(np.abs(np.array(before["probabilities"])-after["probabilities"]).sum()),
                      modeled_claims=sum(_modeled(c) for c in candidate.record["response"]["claims"]),
                      f_executed=True, f_model_sha256=self.selector.f_sha256,
                      outcome=None, realized_gain=None, formal_runtime_executed=False)
        self.trace.append(update)
        return deepcopy(update)


def run_request(selector, request):
    """Single-match, target-free research entry point suitable for local callers.

    Callers supply already acquired/reviewed bodies. This does not acquire them
    or claim that g saved their acquisition/review cost. No result field exists.
    """
    allowed = {"schema", "physical_match_id", "prior", "candidates", "known_title_records",
               "hours_to_kickoff", "evidence_order", "unread_candidate_ids", "model_name",
               "max_updates", "incremental_cost_loss_units"}
    if set(request) != allowed or request["schema"] != SCHEMA:
        raise ValueError("Invalid postread request allowlist/schema")
    candidate_fields = {"candidate_id", "action_id", "body_sha256", "record", "clocks"}
    candidates = []
    for row in request["candidates"]:
        if set(row) != candidate_fields:
            raise ValueError("Candidate request fields differ")
        candidates.append(BodyCandidate.from_record(**row))
    if request["model_name"] not in selector.model_names:
        raise ValueError("Unknown frozen g estimator")
    budget = request["max_updates"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("Nonnegative integer max_updates required")
    session = PostreadSession(selector, physical_match_id=request["physical_match_id"], prior=request["prior"],
        candidates=candidates, known_title_records=request["known_title_records"],
        hours_to_kickoff=request["hours_to_kickoff"], evidence_order=request["evidence_order"],
        unread_candidate_ids=request["unread_candidate_ids"])
    before = session.prediction()
    decisions = []
    while True:
        decision = session.choose(model_name=request["model_name"], remaining_updates=budget-len(session.trace),
                                  incremental_cost=request["incremental_cost_loss_units"])
        decisions.append(decision)
        if decision["selected_candidate_id"] is None:
            break
        session.adopt(decision["selected_candidate_id"])
    return dict(schema=SCHEMA, mode="offline_research", physical_match_id=request["physical_match_id"],
                f_model_sha256=selector.f_sha256, g_model_sha256=selector.g_sha256,
                model_name=request["model_name"], p_before=before["probabilities"],
                p_after=session.prediction()["probabilities"], g_read_body_ids=list(session.candidates),
                f_adopted_body_ids=session.adopted_ids, unread_candidate_ids=list(session.unread_candidate_ids),
                decisions=decisions, f_updates=session.trace, stop_reason=decisions[-1]["stop_reason"],
                source_requests=0, external_api_calls=0, outcome=None, realized_gain=None,
                production_selection_allowed=False, empirical_training_eligible=False,
                cost_scope="Candidate body acquisition and semantic review already incurred; amounts unknown")
